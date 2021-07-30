import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from kornia.filters import spatial_gradient


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.mean.requires_grad = False
        self.std.requires_grad = False
        self.resize = resize

    def forward(self, syn_imgs, gt_imgs):
        syn_imgs = (syn_imgs - self.mean) / self.std
        gt_imgs = (gt_imgs - self.mean) / self.std
        if self.resize:
            syn_imgs = self.transform(syn_imgs, mode="bilinear", size=(224, 224),
                                      align_corners=False)
            gt_imgs = self.transform(gt_imgs, mode="bilinear", size=(224, 224),
                                     align_corners=False)

        loss = 0.0
        x = syn_imgs
        y = gt_imgs
        for block in self.blocks:
            with torch.no_grad():
                x = block(x)
                y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean((1, 2, 3))
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.mean()


def edge_aware_loss(img, disp, gmin, grad_ratio):
    # Compute img grad and grad_max
    grad_img = torch.abs(spatial_gradient(img)).sum(1, keepdim=True).to(torch.float32)
    grad_img_x = grad_img[:, :, 0]
    grad_max_x = torch.amax(grad_img_x, dim=(1, 2, 3), keepdim=True)
    grad_img_y = grad_img[:, :, 1]
    grad_max_y = torch.amax(grad_img_y, dim=(1, 2, 3), keepdim=True)

    # Compute edge mask
    edge_mask_x = grad_img_x / (grad_max_x * grad_ratio)
    edge_mask_y = grad_img_y / (grad_max_y * grad_ratio)
    edge_mask_x = torch.where(edge_mask_x < 1, edge_mask_x, torch.ones_like(edge_mask_x).cuda())
    edge_mask_y = torch.where(edge_mask_y < 1, edge_mask_y, torch.ones_like(edge_mask_y).cuda())

    # Compute and normalize disp grad
    grad_disp = torch.abs(spatial_gradient(disp, normalized=False))
    grad_disp_x = F.instance_norm(grad_disp[:, :, 0])
    grad_disp_y = F.instance_norm(grad_disp[:, :, 1])

    # Compute loss
    grad_disp_x = grad_disp_x - gmin
    grad_disp_y = grad_disp_y - gmin
    loss_map_x = torch.where(grad_disp_x > 0.0, grad_disp_x,
                             torch.zeros_like(grad_disp_x).cuda()) * (1.0 - edge_mask_x)
    loss_map_y = torch.where(grad_disp_y > 0.0, grad_disp_y,
                             torch.zeros_like(grad_disp_y).cuda()) * (1.0 - edge_mask_y)
    return (loss_map_x + loss_map_y).mean()


def edge_aware_loss_v2(img, disp):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    mean_disp = disp.mean(2, True).mean(3, True)
    disp = disp / (mean_disp + 1e-7)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()
