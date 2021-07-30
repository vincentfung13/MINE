import torch

from operations.homography_sampler import HomographySample
from operations.rendering_utils import transform_G_xyz, sample_pdf, gather_pixel_by_pxpy


def render(rgb_BS3HW, sigma_BS1HW, xyz_BS3HW, use_alpha=False, is_bg_depth_inf=False):
    if not use_alpha:
        imgs_syn, depth_syn, blend_weights, weights = plane_volume_rendering(
            rgb_BS3HW,
            sigma_BS1HW,
            xyz_BS3HW,
            is_bg_depth_inf
        )
    else:
        imgs_syn, weights = alpha_composition(sigma_BS1HW, rgb_BS3HW)
        depth_syn, _ = alpha_composition(sigma_BS1HW, xyz_BS3HW[:, :, 2:])
        # No rgb blending with alpha composition
        blend_weights = torch.zeros_like(rgb_BS3HW).cuda()
    return imgs_syn, depth_syn, blend_weights, weights


def alpha_composition(alpha_BK1HW, value_BKCHW):
    """
    composition equation from 'Single-View View Synthesis with Multiplane Images'
    K is the number of planes, k=0 means the nearest plane, k=K-1 means the farthest plane
    :param alpha_BK1HW: alpha at each of the K planes
    :param value_BKCHW: rgb/disparity at each of the K planes
    :return:
    """
    B, K, _, H, W = alpha_BK1HW.size()
    alpha_comp_cumprod = torch.cumprod(1 - alpha_BK1HW, dim=1)  # BxKx1xHxW

    preserve_ratio = torch.cat((torch.ones((B, 1, 1, H, W), dtype=alpha_BK1HW.dtype, device=alpha_BK1HW.device),
                                alpha_comp_cumprod[:, 0:K-1, :, :, :]), dim=1)  # BxKx1xHxW
    weights = alpha_BK1HW * preserve_ratio  # BxKx1xHxW
    value_composed = torch.sum(value_BKCHW * weights, dim=1, keepdim=False)  # Bx3xHxW

    return value_composed, weights


def plane_volume_rendering(rgb_BS3HW, sigma_BS1HW, xyz_BS3HW, is_bg_depth_inf):
    B, S, _, H, W = sigma_BS1HW.size()

    xyz_diff_BS3HW = xyz_BS3HW[:, 1:, :, :, :] - xyz_BS3HW[:, 0:-1, :, :, :]  # Bx(S-1)x3xHxW
    xyz_dist_BS1HW = torch.norm(xyz_diff_BS3HW, dim=2, keepdim=True)  # Bx(S-1)x1xHxW

    xyz_dist_BS1HW = torch.cat((xyz_dist_BS1HW,
                                torch.full((B, 1, 1, H, W),
                                           fill_value=1e3,
                                           dtype=xyz_BS3HW.dtype,
                                           device=xyz_BS3HW.device)),
                               dim=1)  # BxSx3xHxW
    transparency = torch.exp(-sigma_BS1HW * xyz_dist_BS1HW)  # BxSx1xHxW
    alpha = 1 - transparency # BxSx1xHxW

    # add small eps to avoid zero transparency_acc
    # pytorch.cumprod is like: [a, b, c] -> [a, a*b, a*b*c], we need to modify it to [1, a, a*b]
    transparency_acc = torch.cumprod(transparency + 1e-6, dim=1)  # BxSx1xHxW
    transparency_acc = torch.cat((torch.ones((B, 1, 1, H, W), dtype=transparency.dtype, device=transparency.device),
                                  transparency_acc[:, 0:-1, :, :, :]),
                                 dim=1)  # BxSx1xHxW

    weights = transparency_acc * alpha  # BxSx1xHxW
    rgb_out, depth_out = weighted_sum_mpi(rgb_BS3HW, xyz_BS3HW, weights, is_bg_depth_inf)

    return rgb_out, depth_out, transparency_acc, weights


def weighted_sum_mpi(rgb_BS3HW, xyz_BS3HW, weights, is_bg_depth_inf):
    weights_sum = torch.sum(weights, dim=1, keepdim=False)  # Bx1xHxW
    rgb_out = torch.sum(weights * rgb_BS3HW, dim=1, keepdim=False)  # Bx3xHxW

    if is_bg_depth_inf:
        # for dtu dataset, set large depth if weight_sum is small
        depth_out = torch.sum(weights * xyz_BS3HW[:, :, 2:, :, :], dim=1, keepdim=False) \
                    + (1 - weights_sum) * 1000
    else:
        depth_out = torch.sum(weights * xyz_BS3HW[:, :, 2:, :, :], dim=1, keepdim=False) \
                    / (weights_sum + 1e-5)  # Bx1xHxW

    return rgb_out, depth_out


def get_xyz_from_depth(meshgrid_homo,
                       depth,
                       K_inv):
    """

    :param meshgrid_homo: 3xHxW
    :param depth: Bx1xHxW
    :param K_inv: Bx3x3
    :return:
    """
    H, W = meshgrid_homo.size(1), meshgrid_homo.size(2)
    B, _, H_d, W_d = depth.size()
    assert H==H_d, W==W_d

    # 3xHxW -> Bx3xHxW
    meshgrid_src_homo = meshgrid_homo.unsqueeze(0).repeat(B, 1, 1, 1)
    meshgrid_src_homo_B3N = meshgrid_src_homo.reshape(B, 3, -1)
    xyz_src = torch.matmul(K_inv, meshgrid_src_homo_B3N)  # Bx3xHW
    xyz_src = xyz_src.reshape(B, 3, H, W) * depth  # Bx3xHxW

    return xyz_src


def disparity_consistency_src_to_tgt(meshgrid_homo, K_src_inv, disparity_src,
                                     G_tgt_src, K_tgt, disparity_tgt):
    """

    :param xyz_src_B3N: Bx3xN
    :param G_tgt_src: Bx4x4
    :param K_tgt: Bx3x3
    :param disparity_tgt: Bx1xHxW
    :return:
    """
    B, _, H, W = disparity_src.size()
    depth_src = torch.reciprocal(disparity_src)
    xyz_src_B3N = get_xyz_from_depth(meshgrid_homo, depth_src, K_src_inv).view(B, 3, H*W)

    xyz_tgt_B3N = transform_G_xyz(G_tgt_src, xyz_src_B3N, is_return_homo=False)
    K_xyz_tgt_B3N = torch.matmul(K_tgt, xyz_tgt_B3N)
    pxpy_tgt_B2N = K_xyz_tgt_B3N[:, 0:2, :] / K_xyz_tgt_B3N[:, 2:, :]  # Bx2xN

    pxpy_tgt_mask = torch.logical_and(
        torch.logical_and(pxpy_tgt_B2N[:, 0:1, :] >= 0,
                          pxpy_tgt_B2N[:, 0:1, :] <= W - 1),
        torch.logical_and(pxpy_tgt_B2N[:, 1:2, :] >= 0,
                          pxpy_tgt_B2N[:, 1:2, :] <= H - 1)
    )  # B1N

    disparity_src = torch.reciprocal(xyz_tgt_B3N[:, 2:, :])  # Bx1xN
    disparity_tgt = gather_pixel_by_pxpy(disparity_tgt, pxpy_tgt_B2N)  # Bx1xN

    depth_diff = torch.abs(disparity_src - disparity_tgt)
    return torch.mean(depth_diff[pxpy_tgt_mask])


def get_src_xyz_from_plane_disparity(meshgrid_src_homo,
                                     mpi_disparity_src,
                                     K_src_inv):
    """

    :param meshgrid_src_homo: 3xHxW
    :param mpi_disparity_src: BxS
    :param K_src_inv: Bx3x3
    :return:
    """
    B, S = mpi_disparity_src.size()
    H, W = meshgrid_src_homo.size(1), meshgrid_src_homo.size(2)
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).reshape(B * S, 3, 3)

    # 3xHxW -> BxSx3xHxW
    meshgrid_src_homo = meshgrid_src_homo.unsqueeze(0).unsqueeze(1).repeat(B, S, 1, 1, 1)
    meshgrid_src_homo_Bs3N = meshgrid_src_homo.reshape(B * S, 3, -1)
    xyz_src = torch.matmul(K_src_inv_Bs33, meshgrid_src_homo_Bs3N)  # BSx3xHW
    xyz_src = xyz_src.reshape(B, S, 3, H * W) * mpi_depth_src.unsqueeze(2).unsqueeze(3)  # BxSx3xHW
    xyz_src_BS3HW = xyz_src.reshape(B, S, 3, H, W)

    return xyz_src_BS3HW


def get_tgt_xyz_from_plane_disparity(xyz_src_BS3HW,
                                     G_tgt_src):
    """

    :param xyz_src_BS3HW: BxSx3xHxW
    :param G_tgt_src: Bx4x4
    :return:
    """
    B, S, _, H, W = xyz_src_BS3HW.size()
    G_tgt_src_Bs33 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).reshape(B*S, 4, 4)
    xyz_tgt = transform_G_xyz(G_tgt_src_Bs33, xyz_src_BS3HW.reshape(B*S, 3, H*W))  # Bsx3xHW
    xyz_tgt_BS3HW = xyz_tgt.reshape(B, S, 3, H, W)  # BxSx3xHxW
    return xyz_tgt_BS3HW


def render_tgt_rgb_depth(H_sampler: HomographySample,
                         mpi_rgb_src,
                         mpi_sigma_src,
                         mpi_disparity_src,
                         xyz_tgt_BS3HW,
                         G_tgt_src,
                         K_src_inv, K_tgt,
                         use_alpha=False,
                         is_bg_depth_inf=False):
    """
    :param H_sampler:
    :param mpi_rgb_src: BxSx3xHxW
    :param mpi_sigma_src: BxSx1xHxW
    :param mpi_disparity_src: BxS
    :param xyz_tgt_BS3HW: BxSx3xHxW
    :param G_tgt_src: Bx4x4
    :param K_src_inv: Bx3x3
    :param K_tgt: Bx3x3
    :return:
    """
    B, S, _, H, W = mpi_rgb_src.size()
    mpi_depth_src = torch.reciprocal(mpi_disparity_src)  # BxS

    # note that here we concat the mpi_src with xyz_tgt, because H_sampler will sample them for tgt frame
    # mpi_src is the same in whatever frame, but xyz has to be in tgt frame
    mpi_xyz_src = torch.cat((mpi_rgb_src, mpi_sigma_src, xyz_tgt_BS3HW), dim=2)  # BxSx(3+1+3)xHxW

    # homography warping of mpi_src into tgt frame
    G_tgt_src_Bs44 = G_tgt_src.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 4, 4)  # Bsx4x4
    K_src_inv_Bs33 = K_src_inv.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3
    K_tgt_Bs33 = K_tgt.unsqueeze(1).repeat(1, S, 1, 1).contiguous().reshape(B*S, 3, 3)  # Bsx3x3

    # BsxCxHxW, BsxHxW
    tgt_mpi_xyz_BsCHW, tgt_mask_BsHW = H_sampler.sample(mpi_xyz_src.view(B*S, 7, H, W),
                                                        mpi_depth_src.view(B*S),
                                                        G_tgt_src_Bs44,
                                                        K_src_inv_Bs33,
                                                        K_tgt_Bs33)

    # mpi composition
    tgt_mpi_xyz = tgt_mpi_xyz_BsCHW.view(B, S, 7, H, W)
    tgt_rgb_BS3HW = tgt_mpi_xyz[:, :, 0:3, :, :]
    tgt_sigma_BS1HW = tgt_mpi_xyz[:, :, 3:4, :, :]
    tgt_xyz_BS3HW = tgt_mpi_xyz[:, :, 4:, :, :]

    tgt_mask_BSHW = tgt_mask_BsHW.view(B, S, H, W)
    tgt_mask_BSHW = torch.where(tgt_mask_BSHW,
                                torch.ones((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device),
                                torch.zeros((B, S, H, W), dtype=torch.float32, device=mpi_rgb_src.device))

    # Bx3xHxW, Bx1xHxW, Bx1xHxW
    tgt_z_BS1HW = tgt_xyz_BS3HW[:, :, -1:]
    tgt_sigma_BS1HW = torch.where(tgt_z_BS1HW >= 0,
                                  tgt_sigma_BS1HW,
                                  torch.zeros_like(tgt_sigma_BS1HW, device=tgt_sigma_BS1HW.device))
    tgt_rgb_syn, tgt_depth_syn, _, _ = render(tgt_rgb_BS3HW, tgt_sigma_BS1HW, tgt_xyz_BS3HW,
                                              use_alpha=use_alpha,
                                              is_bg_depth_inf=is_bg_depth_inf)
    tgt_mask = torch.sum(tgt_mask_BSHW, dim=1, keepdim=True)  # Bx1xHxW

    return tgt_rgb_syn, tgt_depth_syn, tgt_mask


def predict_mpi_coarse_to_fine(mpi_predictor, src_imgs, xyz_src_BS3HW_coarse,
                               disparity_coarse_src, S_fine, is_bg_depth_inf):
    if S_fine > 0:
        with torch.no_grad():
            # predict coarse mpi
            mpi_coarse_src_list = mpi_predictor(src_imgs, disparity_coarse_src)  # BxS_coarsex4xHxW
            mpi_coarse_rgb_src = mpi_coarse_src_list[0][:, :, 0:3, :, :]  # BxSx1xHxW
            mpi_coarse_sigma_src = mpi_coarse_src_list[0][:, :, 3:, :, :]  # BxSx1xHxW
            _, _, _, weights = plane_volume_rendering(
                mpi_coarse_rgb_src,
                mpi_coarse_sigma_src,
                xyz_src_BS3HW_coarse,
                is_bg_depth_inf
            )
            weights = weights.mean((2, 3, 4)).unsqueeze(1).unsqueeze(2)

            # sample fine disparity
            disparity_fine_src = sample_pdf(disparity_coarse_src.unsqueeze(1).unsqueeze(2), weights, S_fine)
            disparity_fine_src = disparity_fine_src.squeeze(2).squeeze(1)

            # assemble coarse and fine disparity
            disparity_all_src = torch.cat((disparity_coarse_src, disparity_fine_src), dim=1) # Bx(S_coarse + S_fine)
            disparity_all_src, _ = torch.sort(disparity_all_src, dim=1, descending=True)
        mpi_all_src_list = mpi_predictor(src_imgs, disparity_all_src)  # BxS_coarsex4xHxW
        return mpi_all_src_list, disparity_all_src
    else:
        mpi_coarse_src_list = mpi_predictor(src_imgs, disparity_coarse_src)  # BxS_coarsex4xHxW
        return mpi_coarse_src_list, disparity_coarse_src
