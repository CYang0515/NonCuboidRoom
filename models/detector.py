from torch import nn

from models.heads import Heads, HRMerge
from models.hr_cfg import model_cfg
from models.hrnet import HighResolutionNet


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        extra = model_cfg['backbone']['extra']
        self.backbone = HighResolutionNet(extra)
        self.merge = HRMerge()
        self.heads = Heads()
        self.init_weights(pretrained=None)

    def forward(self, x):
        x = self.backbone(x)
        x = self.merge(x)
        x = self.heads(x)
        return x

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)
        self.merge.init_weights()
