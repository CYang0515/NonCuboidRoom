import torch
import torch.nn.functional as F
from torch import nn


class HRMerge(nn.Module):
    def __init__(self,
                 in_channels=(32, 64, 128, 256),
                 out_channels=256,
                 normalize=None):
        super(HRMerge, self).__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=out_channels,
                      kernel_size=1),
        )

        self.fpn_conv = nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            outs.append(F.interpolate(
                inputs[i], scale_factor=2 ** i, mode='bilinear'))
        out = torch.cat(outs, dim=1)

        out = self.reduction_conv(out)
        out = self.relu(out)
        out = self.fpn_conv(out)
        return out


class Heads(nn.Module):
    def __init__(self, in_planes=256, out_planes=64):
        super(Heads, self).__init__()
        self.plane_center = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 3, kernel_size=1)
        )

        self.plane_xy = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 2, kernel_size=1)
        )

        self.plane_wh = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 2, kernel_size=1)
        )

        self.plane_params_pixelwise = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 4, kernel_size=1)
        )

        self.plane_params_instance = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 4, kernel_size=1)
        )

        self.line_region = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 1, kernel_size=1)
        )

        self.line_params = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_planes, 2, kernel_size=1)
        )

    def forward(self, x):
        plane_params_pixelwise = self.plane_params_pixelwise(x)
        plane_center = self.plane_center(x)
        plane_wh = self.plane_wh(x)
        plane_xy = self.plane_xy(x)
        plane_params_instance = self.plane_params_instance(x)

        line_region = self.line_region(x)
        line_params = self.line_params(x)

        out = {
            'plane_center': plane_center,
            'plane_offset': plane_xy,
            'plane_wh': plane_wh,
            'plane_params_pixelwise': plane_params_pixelwise,
            'plane_params_instance': plane_params_instance,
            'line_region': line_region,
            'line_params': line_params,
            'feature': x
        }
        return out
