import numpy as np
import torch
import yaml
from easydict import EasyDict
import argparse
import cv2
import copy
from collections import OrderedDict

from datasets import NYU303, CustomDataset, Structured3D
from models import (ConvertLayout, Detector, DisplayLayout, display2Dseg, Loss,
                    Reconstruction, _validate_colormap, post_process)
from scipy.optimize import linear_sum_assignment


def match_by_Hungarian(gt, pred):
    n = len(gt)
    m = len(pred)
    gt = np.array(gt)
    pred = np.array(pred)
    valid = (gt.sum(0) > 0).sum()
    if m == 0:
        raise IOError
    else:
        gt = gt[:, np.newaxis, :, :]
        pred = pred[np.newaxis, :, :, :]
        cost = np.sum((gt+pred) == 2, axis=(2, 3))  # n*m
        row, col = linear_sum_assignment(-1 * cost)
        inter = cost[row, col].sum()
        PE = inter / valid
        return 1 - PE


def evaluate(gtseg, gtdepth, preseg, predepth, evaluate_2D=True, evaluate_3D=True):
    image_iou, image_pe, merror_edge, rmse, us_rmse = 0, 0, 0, 0, 0
    if evaluate_2D:
        # Parse GT polys
        gt_polys_masks = []
        h, w = gtseg.shape
        gt_polys_edges_mask = np.zeros((h, w))
        edge_thickness = 1
        gt_valid_seg = np.ones((h, w))
        labels = np.unique(gtseg)
        for i, label in enumerate(labels):
            gt_poly_mask = gtseg == label
            if label == -1:
                gt_valid_seg[gt_poly_mask] = 0  # zero pad region
            else:
                contours_, hierarchy = cv2.findContours(gt_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(gt_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                gt_polys_masks.append(gt_poly_mask.astype(np.int32))

        def sortPolyBySize(mask):
            return mask.sum()
        gt_polys_masks.sort(key=sortPolyBySize, reverse=True)

        # Parse predictions
        pred_polys_masks = []
        pred_polys_edges_mask = np.zeros((h, w))
        pre_invalid_seg = np.zeros((h, w))
        labels = np.unique(preseg)
        for i, label in enumerate(labels):
            pred_poly_mask = np.logical_and(preseg == label, gt_valid_seg == 1)
            if pred_poly_mask.sum() == 0:
                continue
            if label == -1:
                # zero pad and infinity region
                pre_invalid_seg[pred_poly_mask] = 1
            else:
                contours_, hierarchy = cv2.findContours(pred_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_SIMPLE
                cv2.polylines(pred_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                pred_polys_masks.append(pred_poly_mask.astype(np.int32))
        if len(pred_polys_masks) == 0.:
            pred_polys_edges_mask[edge_thickness:-
                                  edge_thickness, edge_thickness:-edge_thickness] = 1
            pred_polys_edges_mask = 1 - pred_polys_edges_mask
            pred_poly_mask = np.ones((h, w))
            pred_polys_masks = [pred_poly_mask]

        pred_polys_masks_cand = copy.copy(pred_polys_masks)
        # Assign predictions to ground truth polygons
        ordered_preds = []
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            best_iou_score = 0.3
            best_pred_ind = None
            best_pred_poly_mask = None
            if len(pred_polys_masks_cand) == 0:
                break
            for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand):
                gt_pred_add = gt_poly_mask + pred_poly_mask
                inter = np.equal(gt_pred_add, 2.).sum()
                union = np.greater(gt_pred_add, 0.).sum()
                iou_score = inter / union

                if iou_score > best_iou_score:
                    best_iou_score = iou_score
                    best_pred_ind = pred_ind
                    best_pred_poly_mask = pred_poly_mask
            ordered_preds.append(best_pred_poly_mask)

            pred_polys_masks_cand = [pred_poly_mask for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand)
                                     if pred_ind != best_pred_ind]
            if best_pred_poly_mask is None:
                continue

        ordered_preds += pred_polys_masks_cand
        class_num = max(len(ordered_preds), len(gt_polys_masks))
        colormap = _validate_colormap(None, class_num + 1)

        # Generate GT poly mask
        gt_layout_mask = np.zeros((h, w))
        gt_layout_mask_colored = np.zeros((h, w, 3))
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            gt_layout_mask = np.maximum(
                gt_layout_mask, gt_poly_mask * (gt_ind + 1))
            gt_layout_mask_colored += gt_poly_mask[:,
                                                   :, None] * colormap[gt_ind + 1]

        # Generate pred poly mask
        pred_layout_mask = np.zeros((h, w))
        pred_layout_mask_colored = np.zeros((h, w, 3))
        for pred_ind, pred_poly_mask in enumerate(ordered_preds):
            if pred_poly_mask is not None:
                pred_layout_mask = np.maximum(
                    pred_layout_mask, pred_poly_mask * (pred_ind + 1))
                pred_layout_mask_colored += pred_poly_mask[:,
                                                           :, None] * colormap[pred_ind + 1]

        # Calc IOU
        ious = []
        for layout_comp_ind in range(1, len(gt_polys_masks) + 1):
            inter = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                   np.equal(pred_layout_mask, layout_comp_ind)).sum()
            fp = np.logical_and(np.not_equal(gt_layout_mask, layout_comp_ind),
                                np.equal(pred_layout_mask, layout_comp_ind)).sum()
            fn = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                np.not_equal(pred_layout_mask, layout_comp_ind)).sum()
            union = inter + fp + fn
            iou = inter / union
            ious.append(iou)

        image_iou = sum(ious) / class_num

        # Calc PE
        image_pe = 1 - np.equal(gt_layout_mask[gt_valid_seg == 1],
                                pred_layout_mask[gt_valid_seg == 1]).sum() / (np.sum(gt_valid_seg == 1))
        # Calc PE by Hungarian
        image_pe_hung = match_by_Hungarian(gt_polys_masks, pred_polys_masks)
        # Calc edge error
        # ignore edges at image borders
        img_bound_mask = np.zeros_like(pred_polys_edges_mask)
        img_bound_mask[10:-10, 10:-10] = 1

        pred_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - pred_polys_edges_mask)).astype(np.uint8),
                                                cv2.DIST_L2, 3)
        gt_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - gt_polys_edges_mask)).astype(np.uint8),
                                              cv2.DIST_L2, 3)

        chamfer_dist = pred_polys_edges_mask * gt_dist_trans + \
            gt_polys_edges_mask * pred_dist_trans
        merror_edge = 0.5 * np.sum(chamfer_dist) / np.sum(
            np.greater(img_bound_mask * (gt_polys_edges_mask), 0))

    # Evaluate in 3D
    if evaluate_3D:
        max_depth = 50
        gt_layout_depth_img_mask = np.greater(gtdepth, 0.)
        gt_layout_depth_img = 1. / gtdepth[gt_layout_depth_img_mask]
        gt_layout_depth_img = np.clip(gt_layout_depth_img, 0, max_depth)
        gt_layout_depth_med = np.median(gt_layout_depth_img)
        # max_depth = np.max(gt_layout_depth_img)
        # may be max_depth should be max depth of all scene
        predepth[predepth == 0] = 1 / max_depth
        pred_layout_depth_img = 1. / predepth[gt_layout_depth_img_mask]
        pred_layout_depth_img = np.clip(pred_layout_depth_img, 0, max_depth)
        pred_layout_depth_med = np.median(pred_layout_depth_img)

        # Calc MSE
        ms_error_image = (pred_layout_depth_img - gt_layout_depth_img) ** 2
        rmse = np.sqrt(np.sum(ms_error_image) /
                       np.sum(gt_layout_depth_img_mask))

        # Calc up to scale MSE
        if np.isnan(pred_layout_depth_med) or pred_layout_depth_med == 0:
            d_scale = 1.
        else:
            d_scale = gt_layout_depth_med / pred_layout_depth_med
        us_ms_error_image = (
            d_scale * pred_layout_depth_img - gt_layout_depth_img) ** 2
        us_rmse = np.sqrt(np.sum(us_ms_error_image) /
                          np.sum(gt_layout_depth_img_mask))

    return image_iou, image_pe, merror_edge, rmse, us_rmse, image_pe_hung


def test_structured3d(model, criterion, dataloader, device, cfg):
    model.eval()
    results = []
    for iters, inputs in enumerate(dataloader):
        print(f'{iters}/{len(dataloader)}')
        # set device
        for key, value in inputs.items():
            inputs[key] = value.to(device)

        # forward
        x = model(inputs['img'])
        loss, loss_stats = criterion(x, **inputs)

        # post process on output feature map size and extract plane and line detection results
        dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise = post_process(x, Mnms=1)

        for i in range(1):
            # generate layout with a post-process according to detection results
            (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), (pfloor, pceiling) = Reconstruction(
                dt_planes[i],
                dt_params3d_instance[i],
                dt_lines[i],
                K=inputs['intri'][i].cpu().numpy(),
                size=(720, 1280),
                threshold=(0.3, 0.3, 0.3, 0.3))

            # convert no opt results to segmentation and depth map and evaluate results
            _seg, _depth, _, _polys = ConvertLayout(
                inputs['img'][i], _ups, _downs, _attribution,
                K=inputs['intri'][i].cpu().numpy(), pwalls=_params_layout,
                pfloor=pfloor, pceiling=pceiling,
                ixy1map=inputs['ixy1map'][i].cpu().numpy(),
                valid=inputs['iseg'][i].cpu().numpy(),
                oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)

            _res = evaluate(inputs['iseg'][i].cpu().numpy(),
                            inputs['idepth'][i].cpu().numpy(), _seg, _depth)

            # convert opt results to segmentation and depth map and evaluate results
            seg, depth, img, polys = ConvertLayout(
                inputs['img'][i], ups, downs, attribution,
                K=inputs['intri'][i].cpu().numpy(), pwalls=params_layout,
                pfloor=pfloor, pceiling=pceiling,
                ixy1map=inputs['ixy1map'][i].cpu().numpy(),
                valid=inputs['iseg'][i].cpu().numpy(),
                oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)

            res = evaluate(inputs['iseg'][i].cpu().numpy(),
                           inputs['idepth'][i].cpu().numpy(), seg, depth)

            # print metric results
            results.append([_res, res])
            print(np.mean(np.array(results), axis=0))

            if cfg.visual:
                # display layout
                DisplayLayout(img, seg, depth, polys, _seg, _depth, _polys, inputs['iseg'][i].cpu(
                ).numpy(), inputs['ilbox'][i].cpu().numpy(), iters)


def test_nyu303(model, criterion, dataloader, device, cfg):
    model.eval()
    results = []
    for iters, inputs in enumerate(dataloader):
        print(f'{iters}/{len(dataloader)}')
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        # forward
        x = model(inputs['img'])
        loss, loss_stats = criterion(x)

        # post process on output feature map size
        dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise = post_process(x, Mnms=1)
        # convert sunrgbd crop image to fullres img
        dt_planes[:, :, :4] = dt_planes[:, :, :4] + \
            np.array([41, 45, 41, 45]) / 4.
        dt_lines[:, :, 1] = dt_lines[:, :, 1] + \
            41/4. - dt_lines[:, :, 0] * 45/4.

        # reconstruction
        for i in range(1):
            (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), (pfloor, pceiling) = Reconstruction(
                dt_planes[i], 
                dt_params3d_instance[i], 
                dt_lines[i],
                K=inputs['full_intri'][i].cpu().numpy(), 
                size=(480, 640), 
                threshold=(0.12, 0.1, 0.1, 0.2), 
                downsample=4)

            # no opt
            _seg, _depth, img, _ = ConvertLayout(inputs['fullimg'][i], _ups, _downs, _attribution,
                                                 K=inputs['full_intri'][i].cpu().numpy(), pwalls=_params_layout,
                                                 pfloor=pfloor, pceiling=pceiling,
                                                 ixy1map=inputs['ixy1map'][i].cpu().numpy(), valid=inputs['iseg'][i].cpu().numpy())
            _res = evaluate(inputs['iseg'][i].cpu().numpy(), inputs['idepth'][i].cpu().numpy(), _seg, _depth)
            
            # opt
            seg, depth, _, _ = ConvertLayout(inputs['fullimg'][i], ups, downs, attribution,
                                             K=inputs['full_intri'][i].cpu().numpy(), pwalls=params_layout,
                                             pfloor=pfloor, pceiling=pceiling,
                                             ixy1map=inputs['ixy1map'][i].cpu().numpy(), valid=inputs['iseg'][i].cpu().numpy())
            res = evaluate(inputs['iseg'][i].cpu().numpy(), inputs['idepth'][i].cpu().numpy(), seg, depth)

            results.append([_res, res])
            print(np.mean(np.array(results), axis=0)[:,-1])
            
            if cfg.visual:
                display2Dseg(img=inputs['fullimg'][i], segs_pred=seg, segs_gt=inputs['iseg'][i].cpu().numpy(), label=inputs['ilbox'][0].cpu().numpy(),
                            iters=f'{iters}', method='opt_nyu303', draw_gt=1)
            if cfg.exam:
                return


def test_custom(model, criterion, dataloader, device, cfg):
    model.eval()
    for iters, inputs in enumerate(dataloader):
        print(f'{iters}/{len(dataloader)}')
        # set device
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        # forward
        x = model(inputs['img'])
        loss, loss_stats = criterion(x)

        # post process on output feature map size, and extract planes, lines, plane params instance and plane params pixelwise
        dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise = post_process(x, Mnms=1)

        # reconstruction according to detection results
        for i in range(1):
            (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), ( pfloor, pceiling) = Reconstruction(
                dt_planes[i],
                dt_params3d_instance[i],
                dt_lines[i],
                K=inputs['intri'][i].cpu().numpy(),
                size=(720, 1280),
                threshold=(0.3, 0.05, 0.05, 0.3))

            # convert intersection points to segmentation for visual
            # no opt results
            _seg, _depth, _, _polys = ConvertLayout(
                inputs['img'][i], _ups, _downs, _attribution, K=inputs['intri'][i].cpu().numpy(),
                pwalls=_params_layout, pfloor=pfloor, pceiling=pceiling,
                ixy1map=inputs['ixy1map'][i].cpu().numpy(),
                valid=inputs['iseg'][i].cpu().numpy(),
                oxy1map=inputs['oxy1map'][i].cpu().numpy(),
                pixelwise=None
            )
            # opt results
            seg, depth, img, polys = ConvertLayout(
                inputs['img'][i], ups, downs, attribution, K=inputs['intri'][i].cpu().numpy(),
                pwalls=params_layout, pfloor=pfloor, pceiling=pceiling,
                ixy1map=inputs['ixy1map'][i].cpu().numpy(),
                valid=inputs['iseg'][i].cpu().numpy(),
                oxy1map=inputs['oxy1map'][i].cpu().numpy(),
                pixelwise=None
            )

            if cfg.visual:
                # display layout
                DisplayLayout(img, seg, depth, polys, _seg, _depth, _polys, inputs['iseg'][i].cpu().numpy(),
                    inputs['ilbox'][i].cpu().numpy(), iters)

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='Structured3D', choices=['Structured3D', 'NYU303', 'CUSTOM'])
    parser.add_argument('--pretrained', type=str, default=None, required=True, help='the pretrained model')
    parser.add_argument('--visual', action='store_true', help='whether to visual the results')
    parser.add_argument('--exam', action='store_true', help='test one example on nyu303 dataset')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with open('cfg.yaml', 'r') as f:
        config = yaml.load(f)
        cfg = EasyDict(config)
    args = parse()
    cfg.update(vars(args))

    if cfg.exam:
        assert cfg.data == 'NYU303', 'provide one example of nyu303 to test'
    #  dataset
    if cfg.data == 'Structured3D':
        dataset = Structured3D(cfg.Dataset.Structured3D, 'test')
    elif cfg.data == 'NYU303':
        dataset = NYU303(cfg.Dataset.NYU303, 'test', exam=cfg.exam)
    elif cfg.data == 'CUSTOM':
        dataset = CustomDataset(cfg.Dataset.CUSTOM, 'test')
    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=cfg.num_workers)

    # create network
    model = Detector()
    # compute loss
    criterion = Loss(cfg.Weights)

    # set data parallel
    # if cfg.num_gpus > 1 and torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model)

    # reload weights
    if cfg.pretrained:
        state_dict = torch.load(cfg.pretrained,
                                map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    if cfg.data == 'Structured3D':
        test_structured3d(model, criterion, dataloader, device, cfg)
    elif cfg.data == 'NYU303':
        test_nyu303(model, criterion, dataloader, device, cfg)
    elif cfg.data == 'CUSTOM':
        test_custom(model, criterion, dataloader, device, cfg)
    else:
        raise NotImplementedError
