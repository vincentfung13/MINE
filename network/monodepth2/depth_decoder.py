# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.monodepth2.layers import *


def conv(in_planes, out_planes, kernel_size, instancenorm=False):
    if instancenorm:
        m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    return m


class DepthDecoder(nn.Module):
    def tuple_to_str(self, key_tuple):
        key_str = '-'.join(str(key_tuple))
        return key_str

    def __init__(self, num_ch_enc, embedder, embedder_out_dim,
                 use_alpha=False, scales=range(4), num_output_channels=4,
                 use_skips=True, sigma_dropout_rate=0.0, **kwargs):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.use_alpha = use_alpha
        self.sigma_dropout_rate = sigma_dropout_rate

        self.embedder = embedder
        self.E = embedder_out_dim

        final_enc_out_channels = num_ch_enc[-1]
        self.downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_down1 = conv(final_enc_out_channels, 512, 1, False)
        self.conv_down2 = conv(512, 256, 3, False)
        self.conv_up1 = conv(256, 256, 3, False)
        self.conv_up2 = conv(256, final_enc_out_channels, 1, False)

        self.num_ch_enc = num_ch_enc
        print("num_ch_enc=", num_ch_enc)
        self.num_ch_enc = [x + self.E for x in self.num_ch_enc]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        # self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[self.tuple_to_str(("upconv", i, 0))] = ConvBlock(num_ch_in, num_ch_out)
            print("upconv_{}_{}".format(i, 0), num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[self.tuple_to_str(("upconv", i, 1))] = ConvBlock(num_ch_in, num_ch_out)
            print("upconv_{}_{}".format(i, 1), num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[self.tuple_to_str(("dispconv", s))] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)


        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, disparity):
        B, S = disparity.size()
        disparity = self.embedder(disparity.reshape(B * S, 1)).unsqueeze(2).unsqueeze(3)

        # extension of encoder to increase receptive field
        encoder_out = input_features[-1]
        conv_down1 = self.conv_down1(self.downsample(encoder_out))
        conv_down2 = self.conv_down2(self.downsample(conv_down1))
        conv_up1 = self.conv_up1(self.upsample(conv_down2))
        conv_up2 = self.conv_up2(self.upsample(conv_up1))

        # repeat / reshape features
        _, C_feat, H_feat, W_feat = conv_up2.size()
        feat_tmp = conv_up2.unsqueeze(1).expand(B, S, C_feat, H_feat, W_feat) \
            .contiguous().view(B * S, C_feat, H_feat, W_feat)
        disparity_BsCHW = disparity.repeat(1, 1, H_feat, W_feat)
        conv_up2 = torch.cat((feat_tmp, disparity_BsCHW), dim=1)

        # repeat / reshape features
        for i, feat in enumerate(input_features):
            _, C_feat, H_feat, W_feat = feat.size()
            feat_tmp = feat.unsqueeze(1).expand(B, S, C_feat, H_feat, W_feat) \
                .contiguous().view(B * S, C_feat, H_feat, W_feat)
            disparity_BsCHW = disparity.repeat(1, 1, H_feat, W_feat)
            input_features[i] = torch.cat((feat_tmp, disparity_BsCHW), dim=1)

        # for i, feat in enumerate(input_features):
        #     _, C_feat, H_feat, W_feat = feat.size()
        #     input_features[i] = feat.unsqueeze(1).expand(B, S, C_feat, H_feat, W_feat) \
        #         .contiguous().view(B * S, C_feat, H_feat, W_feat)

        # decoder
        outputs = {}
        # x = input_features[-1]
        x = conv_up2
        for i in range(4, -1, -1):
            x = self.convs[self.tuple_to_str(("upconv", i, 0))](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[self.tuple_to_str(("upconv", i, 1))](x)
            if i in self.scales:
                output = self.convs[self.tuple_to_str(("dispconv", i))](x)
                H_mpi, W_mpi = output.size(2), output.size(3)
                mpi = output.view(B, S, 4, H_mpi, W_mpi)
                mpi_rgb = self.sigmoid(mpi[:, :, 0:3, :, :])
                mpi_sigma = torch.abs(mpi[:, :, 3:, :, :]) + 1e-4 \
                        if not self.use_alpha \
                        else self.sigmoid(mpi[:, :, 3:, :, :])

                if self.sigma_dropout_rate > 0.0 and self.training:
                    mpi_sigma = F.dropout2d(mpi_sigma, p=self.sigma_dropout_rate)

                outputs[("disp", i)] = torch.cat((mpi_rgb, mpi_sigma), dim=2)

        return outputs
