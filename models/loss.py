import torch
from torch import nn
from torch.nn import functional as F


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _neg_loss(pred, gt, alpha=2, beta=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                               alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, alpha=2, beta=4):
        return self.neg_loss(out, target, alpha, beta)


class RegL1Loss(nn.Module):  # L1 loss
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target, smooth=0):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        if not smooth:
            loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        else:
            loss = F.smooth_l1_loss(
                pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class ParseL1loss(nn.Module):
    def __init__(self):
        super(ParseL1loss, self).__init__()

    def forward(self, output, target, mask):
        mask = (mask == 1).float()
        loss = F.l1_loss(output * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class DepthLoss(nn.Module):
    def __init__(self, pixelwise=True):
        super(DepthLoss, self).__init__()
        self.pixelwise = pixelwise

    def InverDepthByPixelWise(self, output, K_inv, xy1):
        depth = self.InverDepthByND(output, K_inv, xy1)
        return depth

    def InverDepthByND(self, output, K_inv, xy1):
        b, h, w = xy1.size(0), xy1.size(1), xy1.size(2)
        n_d = output[:, :3] / torch.clamp(output[:, 3:], 1e-5, 1e5)  # n/d
        n_d = n_d.view([b, 3, -1])  # b*3*n
        xy1 = xy1.permute([0, 3, 1, 2]).view([b, 3, -1])  # b*3*n
        Qray = torch.matmul(K_inv, xy1)  # b*3*n
        res = -1 * (n_d * Qray).sum(1)
        res = res.view([b, h, w])
        return res

    def InverDepthByInstance(self, output, mask, ind, seg, K_inv, xy1):
        b = seg.size(0)
        output = _transpose_and_gather_feat(output, ind)  # B*N*4
        inversedepth = []
        for i in range(b):
            # for sunrgbd
            valid = torch.cumsum(torch.flip(mask[i], dims=(0,)), dim=0)
            valid = torch.flip(valid, dims=(0,)).gt(0)
            nd = output[i][valid]
            # nd = output[i][mask[i] == 1]
            n_d = (nd[:, :3] / torch.clamp(nd[:, 3:],
                                           1e-5, 1e5)).view([-1, 1, 1, 3])
            K_inv_ = K_inv[i:i+1].view([-1, 1, 3, 3])
            res = -1 * torch.mul(torch.matmul(n_d, K_inv_), xy1[i:i+1]).sum(3)
            index = seg[i:i+1].clone()
            index[index == -1] = 0
            res = torch.gather(res, index=index.long(), dim=0)
            inversedepth.append(res)
        inversedepth = torch.cat(inversedepth, dim=0)
        return inversedepth

    def forward(self, output, target, K, xy1, mask=None, ind=None, seg=None):
        if self.pixelwise:
            depth = self.InverDepthByPixelWise(output, K, xy1)  # 1/z
        else:
            depth = self.InverDepthByInstance(output, mask, ind, seg, K, xy1)

        # ignore insufficent region
        depth = torch.clamp(depth, 0.05, 2)  # 1/z  0.5<z<20
        valid = torch.logical_and(seg.ne(-1), target.gt(0.05))
        valid = torch.logical_and(valid, target.lt(2))
        if valid.sum().float() == 0:
            loss = torch.tensor(0.).float().to(depth.device)
        else:
            loss = F.l1_loss(depth[valid], target[valid])
        return loss


class Loss(torch.nn.Module):
    def __init__(self, cfg, factor=1):
        super(Loss, self).__init__()
        self.crit_cls = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.dense_reg = torch.nn.SmoothL1Loss()
        self.parse_reg = ParseL1loss()
        self.pixelwisedepth = DepthLoss(pixelwise=True)
        self.instancedepth = DepthLoss(pixelwise=False)

        self.w_pcenter = cfg.w_pcenter * factor
        self.w_poffset = cfg.w_poffset * factor
        self.w_psize = cfg.w_psize * factor

        self.w_loffset = cfg.w_loffset * factor
        self.w_lalpha = cfg.w_lalpha * factor
        self.w_lregion = cfg.w_lregion * factor

        self.w_pparam = cfg.w_pparam * factor
        self.w_pparam_i = cfg.w_pparam_i * factor

        self.w_pwdepth = cfg.w_pwdepth * factor
        self.w_insdepth = cfg.w_insdepth * factor

        self.falpha_p = cfg.falpha_p
        self.fbeta_p = cfg.fbeta_p
        self.falpha_l = cfg.falpha_l
        self.fbeta_l = cfg.fbeta_l

    def forward(self, outputs, **batch):
        plane_hm_loss, plane_wh_loss, plane_offset_loss = 0, 0, 0
        plane_param_loss, plane_param_instance_loss = 0, 0
        plane_pixelwise_depth_loss, plane_instance_depth_loss = 0, 0
        line_hm_loss, line_offset_loss, line_alpha_loss = 0, 0, 0

        plane_center = outputs['plane_center']
        plane_wh = outputs['plane_wh']
        plane_offset = outputs['plane_offset']
        plane_params = outputs['plane_params_pixelwise']
        plane_params_instance = outputs['plane_params_instance']

        line_region = outputs['line_region']
        line_params = outputs['line_params']
        line_offset = line_params[:, 0:1]
        line_alpha = line_params[:, 1:2]

        line_alpha = _sigmoid(line_alpha)
        plane_center = _sigmoid(plane_center)
        line_region = _sigmoid(line_region)
        if len(batch) == 0:
            return None, None
        # plane detection loss
        plane_hm_loss += self.crit_cls(plane_center,
                                       batch['plane_hm'], self.falpha_p, self.fbeta_p)
        plane_wh_loss += self.crit_reg(plane_wh,
                                       batch['reg_mask'], batch['ind'], batch['plane_wh'])
        plane_offset_loss += self.crit_reg(
            plane_offset, batch['reg_mask'], batch['ind'], batch['plane_offset'])

        # plane param loss
        valid = batch['oseg'][:, None, :, :].repeat([1, 4, 1, 1]).ne(-1)
        plane_param_loss += self.dense_reg(
            plane_params[valid], batch['plane_params'][valid])
        plane_param_instance_loss += self.crit_reg(
            plane_params_instance, batch['reg_mask'], batch['ind'], batch['params3d'], smooth=1)

        # plane depth loss
        plane_pixelwise_depth_loss += self.pixelwisedepth(
            plane_params, batch['odepth'], batch['intri_inv'], batch['oxy1map'], seg=batch['oseg'])
        plane_instance_depth_loss += self.instancedepth(plane_params_instance, batch['odepth'], batch['intri_inv'], batch['oxy1map'],
                                                        batch['reg_mask'], batch['ind'], batch['oseg'])

        # line detection loss
        line_hm_loss += self.crit_cls(line_region,
                                      batch['line_hm'], self.falpha_l, self.fbeta_l)
        line_offset_loss += self.parse_reg(line_offset,
                                           batch['line_offset'], batch['line_hm'])
        line_alpha_loss += self.parse_reg(line_alpha,
                                          batch['line_alpha'], batch['line_hm'])

        loss = self.w_pcenter * plane_hm_loss + self.w_psize * plane_wh_loss + \
            self.w_poffset * plane_offset_loss + self.w_pparam * plane_param_loss + \
            self.w_pparam_i * plane_param_instance_loss + \
            self.w_pwdepth * plane_pixelwise_depth_loss + \
            self.w_insdepth * plane_instance_depth_loss + \
            self.w_lregion * line_hm_loss + self.w_loffset * line_offset_loss + \
            self.w_lalpha * line_alpha_loss
        loss_stats = {'loss': loss, 'plane_hm_loss': self.w_pcenter * plane_hm_loss,
                      'plane_wh_loss': self.w_psize * plane_wh_loss,
                      'plane_offset_loss': self.w_poffset * plane_offset_loss,
                      'plane_param_loss': self.w_pparam * plane_param_loss,
                      'plane_param_i_loss': self.w_pparam_i * plane_param_instance_loss,
                      'plane_pixelwise_depth_loss': self.w_pwdepth * plane_pixelwise_depth_loss,
                      'plane_instance_depth_loss': self.w_insdepth * plane_instance_depth_loss,
                      'line_hm_loss': self.w_lregion * line_hm_loss,
                      'line_offset_loss': self.w_loffset * line_offset_loss,
                      'line_alpha_loss': self.w_lalpha * line_alpha_loss}
        return loss, loss_stats
