import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size):
    m = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                    stride=1, padding=(kernel_size - 1) // 2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True)
    )
    return m


class VDRPredictor(nn.Module):
    def __init__(self, hidden=16):
        super(VDRPredictor, self).__init__()

        self.conv1 = conv(3 + 3, hidden, 7)
        self.conv2 = conv(hidden, hidden, 3)
        self.conv3 = conv(hidden, hidden, 3)
        self.conv4 = nn.Conv2d(hidden, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.out = nn.Sigmoid()

    def forward(self, rgb_BN3HW, vd_B3HW):
        B, N, _, H, W = rgb_BN3HW.size()

        rgb_Bn3HW = rgb_BN3HW.view(B * N, 3, H, W)

        # Repeat viewing direction and normalize it to unit vectors
        vd_Bn3HW = vd_B3HW.unsqueeze(1).repeat(1, N, 1, 1, 1).view(B * N, 3, H, W)
        vd_Bn3HW = F.normalize(vd_Bn3HW, dim=1)

        # Concat as input
        input_Bn6HW = torch.cat((rgb_Bn3HW, vd_Bn3HW), dim=1)

        # Use the network to produce view dependent radiance
        conv1 = self.conv1(input_Bn6HW)
        conv2 = self.conv2(conv1) + conv1
        conv3 = self.conv3(conv2) + conv2
        vd_rgb_BN3HW = self.out(self.conv4(conv3).view(B, N, 3, H, W))

        # return vd_rgb_BN3HW
        return rgb_BN3HW
