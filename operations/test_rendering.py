import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from operations import mpi_rendering
from operations.homography_sampler import HomographySample


def test_mpi_composition():
    B = 1
    car_np = cv2.imread('/data00/home/jiaxinli/repos/ad-img-to-video/py/images/car.jpg')
    car_np = cv2.cvtColor(car_np, cv2.COLOR_BGR2RGB)
    H, W = car_np.shape[0], car_np.shape[1]
    car = (torch.from_numpy(car_np).to(dtype=torch.float32) / 255).permute(2, 0, 1).contiguous().unsqueeze(0)  # Bx3xHxW
    car = torch.cat((car,
                     torch.ones((1, 1, H, W), dtype=torch.float32, device=car.device)),
                    dim=1)  # Bx4xHxW

    car_2 = car * 0.5

    img0 = torch.ones_like(car)
    img0[:, 3, :, :] = 0.5

    img1 = torch.clone(car)
    img1[:, 3, int(H*0.25):int(H*0.75), int(W*0.25):int(W*0.75)] = 0

    img2 = torch.clone(car_2)

    # build mpi and compose
    mpi = torch.stack([img0, img1, img2], dim=1)  # BxSx4xHxW
    rgb_composed, weights = mpi_rendering.mpi_composition(mpi[:, :, 3:, :, :], mpi[:, :, 0:3, : :])

    rgb_composed_np = rgb_composed.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # BxHxWx3
    vis_np = rgb_composed_np[0, ...]
    plt.figure()
    plt.imshow(vis_np)
    plt.show()


def euler_to_rotation_matrix(x_angle, y_angle, z_angle, seq='xyz', degrees=False):
    """
    Note that here we want to return a rotation matrix rot_mtx, which transform the tgt points into src frame,
    i.e, rot_mtx * p_tgt = p_src
    Therefore we need to add negative to x/y/z_angle
    :param roll:
    :param pitch:
    :param yaw:
    :return:
    """
    r = Rotation.from_euler(seq,
                            [-x_angle, -y_angle, -z_angle],
                            degrees=degrees)
    rot_mtx = r.as_matrix().astype(np.float32)
    return rot_mtx


def rotation_test():
    rx = euler_to_rotation_matrix(0, 0, 30,
                                  seq='xyz',
                                  degrees=True)
    ry = euler_to_rotation_matrix(0, 30, 0,
                                  seq='xyz',
                                  degrees=True)
    rz = euler_to_rotation_matrix(30, 0, 0,
                                  seq='xyz',
                                  degrees=True)

    rxyz = euler_to_rotation_matrix(30, 30, 30,
                                    seq='xyz',
                                    degrees=True)
    print(rxyz)
    print(np.dot(rx, np.dot(ry, rz)))
    print(np.dot(rz, np.dot(ry, rx)))


def K_from_img_HW(B, H, W, device):
    # relationship between FoV and focal length:
    # assume W > H
    # W / 2 = f*tan(\theta / 2)
    # here we default the horizontal FoV as 53.13 degree
    # the vertical FoV can be computed as H/2 = W*tan(\theta/2)
    f = max(H, W)
    K = torch.tensor([[f, 0, W * 0.5],
                      [0, f, H * 0.5],
                      [0, 0, 1]],
                     dtype=torch.float32,
                     device=device).unsqueeze(0).expand(B, 3, 3)
    return


def test_homography_sample():
    car_np = cv2.imread('/data00/home/jiaxinli/repos/ad-img-to-video/py/images/car.jpg')
    car_np = cv2.cvtColor(car_np, cv2.COLOR_BGR2RGB)
    H, W = car_np.shape[0], car_np.shape[1]
    car = (torch.from_numpy(car_np).to(dtype=torch.float32) / 255).permute(2, 0, 1).contiguous().unsqueeze(0)  # Bx3xHxW
    car = torch.cat((car,
                     torch.ones((1, 1, H, W), dtype=torch.float32, device=car.device)),
                    dim=1)  # Bx4xHxW
    img = car
    B = img.size(0)
    d_img = torch.ones((B), dtype=torch.float32, device=img.device)

    homography_sampler = HomographySample(H, W, img.device)
    homography_sampler.sample(car, )


if __name__ == '__main__':
    test_mpi_composition()
