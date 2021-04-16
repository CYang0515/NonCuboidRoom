import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torchvision.transforms as tf
from models.utils import draw_umich_gaussian, gaussian_radius, line_gaussian
from PIL import Image
from shapely.geometry import Polygon
from torch.utils import data 


class SUNRGBD(data.Dataset):
    def __init__(self, config, phase='train', split='all'):
        self.config = config
        self.phase = phase
        self.split = split
        self.max_objs = config.max_objs

        self.colorjitter = tf.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        self.transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if os.path.isfile(f'data/SUNRGBD/sunrgb_s3d_{phase}.json'):
            with open(f'data/SUNRGBD/sunrgb_s3d_{phase}.json') as f:
                self.anno = json.load(f)
        else:
            self.anno = self.convert_suntos3d()
            with open(f'data/SUNRGBD/sunrgb_s3d_{phase}.json', 'w') as f:
                json.dump(self.anno, f)

        # extract NYU dataset
        self.anno_nyu = []
        self.anno_other = []
        for i in self.anno:
            im_name = i['file_name']
            cato = im_name.split('/')[4]
            if cato == 'NYUdata':
                self.anno_nyu.append(i)
            else:
                self.anno_other.append(i)
        self.anno = {'all': self.anno,
                     'nyu': self.anno_nyu, 'other': self.anno_other}

    def __getitem__(self, index):
        sample = self.anno[self.split][index]
        img_name = sample['file_name']
        intri_name = os.path.join(*img_name.split('/')[:-2], 'intrinsics.txt')
        with open(intri_name, 'r') as f:
            K = [[float(x) for x in y.rstrip(' ').lstrip(' ').split(' ')]
                 for y in f.readlines()]
            K = np.array(K).reshape([3, 3])
            self.K = np.array(K).astype(np.float32)
            self.K_inv = np.linalg.inv(K).astype(np.float32)
        img = Image.open(img_name)  # RGB
        if self.phase == 'train' and self.config.colorjitter:
            img = self.colorjitter(img)
        img = np.array(img)
        img, inh, inw = self.padimage(img)
        img = self.transforms(img)

        layout = sample['layout']

        pparams = []
        labels = []
        segs = -1 * np.ones([inh, inw])
        i = 0
        endpoints = []
        for _, pp in enumerate(layout):
            if pp['category'] == 1:  # wall
                polygon = Polygon(np.array(pp['polygon'][0]))
                area = polygon.area
                if area > 1000:
                    cout = np.array(pp['polygon'][0]).astype(np.int32)
                    cv2.fillPoly(segs, [cout], color=i)
                    pparams.append([*pp['plane_param']])
                    labels.append(0)
                    i = i + 1
            else:
                for v in pp['polygon']:
                    cout = np.array(v)
                    if len(cout) <= 2:
                        continue
                    polygon = Polygon(cout)
                    if polygon.area > 1000:
                        cout = cout.astype(np.int32)
                        cv2.fillPoly(segs, [cout], color=i)
                        pparams.append([*pp['plane_param']])
                        if pp['category'] == 2:  # floor
                            labels.append(1)
                        else:
                            labels.append(2)
                        i = i + 1

            if pp['attr'] == 1 or pp['attr'] == 2:  # oc line or intersection line
                xy = np.array(pp['endpoints'])
                if len(xy) > 0:
                    endpoints.append([xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1]])

        # plane detection gt and instance plane params  #center map, wh, offset, instance param
        oh, ow = inh // self.config.downsample, inw // self.config.downsample
        hm = np.zeros((3, oh, ow), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        params3d = np.zeros((self.max_objs, 4), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        for i, (label, param) in enumerate(zip(labels, pparams)):
            yx = np.where(segs == i)
            if len(yx[0]) == 0:
                continue
            box = np.array([np.min(yx[1]), np.min(yx[0]), np.max(
                yx[1]), np.max(yx[0])], dtype=np.float32)
            box /= self.config.downsample
            h = box[3] - box[1]
            w = box[2] - box[0]
            radius = gaussian_radius((np.ceil(h), np.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array(
                [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[label], ct_int, radius)
            wh[i] = 1. * w, 1. * h
            ind[i] = ct_int[1] * ow + ct_int[0]
            reg[i] = ct - ct_int
            reg_mask[i] = 1
            params3d[i, :3] = param[:3]
            params3d[i, 3] = param[3]  # 1. / param[3]

        ret = {'img': img, 'plane_hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'plane_wh': wh,
               'plane_offset': reg,
               'params3d': params3d}

        # line detection gt  # line map, alpha, offset map
        line_hm = np.zeros((3, oh, ow), dtype=np.float32)
        for line in endpoints:
            line = np.array(line) / self.config.downsample
            line = np.reshape(line, [2, 2])
            line_gaussian(line_hm, line, 2)
        ret['line_hm'] = line_hm[0:1]
        ret['line_alpha'] = line_hm[1:2]
        ret['line_offset'] = line_hm[2:3]

        # plane param map gt  # pixel-wise plane param
        plane_params = np.zeros((4, oh, ow), dtype=np.float32)
        plane_params_input = np.zeros((4, inh, inw), dtype=np.float32)
        oseg = cv2.resize(segs, (ow, oh), interpolation=cv2.INTER_NEAREST)
        for i, param in enumerate(pparams):
            param = np.array(param)
            plane_params[:3, oseg == i] = param[:3, np.newaxis]  # normal
            # 1. / param[3]  # offset 1/d
            plane_params[3, oseg == i] = param[3]
            plane_params_input[:3, segs == i] = param[:3, np.newaxis]
            plane_params_input[3, segs == i] = param[3]
        ret['plane_params'] = plane_params

        # param params for depth loss
        # coordinate map
        x = np.arange(ow * 4)
        y = np.arange(oh * 4)
        xx, yy = np.meshgrid(x, y)
        xymap = np.stack([xx, yy], axis=2).astype(np.float32)
        oxymap = cv2.resize(xymap, (ow, oh), interpolation=cv2.INTER_LINEAR)
        oxy1map = np.concatenate([oxymap, np.ones_like(
            oxymap[:, :, :1])], axis=-1).astype(np.float32)
        inverdepth = self.inverdepth(plane_params, self.K_inv, oxy1map)
        # depthmap = cv2.resize(cv2.imread(os.path.join(dirs, 'depth.png'), cv2.IMREAD_UNCHANGED), (ow, oh), interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite('./depth.png', 1/inverdepth*100)
        ret['odepth'] = inverdepth
        ret['oseg'] = oseg
        ret['oxy1map'] = oxy1map

        # reconstructure  # camera intri, plane label, segs, depth
        ret['intri'] = self.K  # np.array(anno['intri']).T
        ret['intri_inv'] = self.K_inv

        # evaluate gt
        ret['iseg'] = segs
        ixymap = cv2.resize(xymap, (inw, inh), interpolation=cv2.INTER_LINEAR)
        ixy1map = np.concatenate([ixymap, np.ones_like(
            ixymap[:, :, :1])], axis=-1).astype(np.float32)
        inverdepth_input = self.inverdepth(
            plane_params_input, self.K_inv, ixy1map)
        ret['ixy1map'] = ixy1map
        ret['idepth'] = inverdepth_input

        return ret

    def __len__(self):
        return len(self.anno[self.split])

    def padimage(self, image):
        outsize = [480, 640, 3]
        h, w = image.shape[0], image.shape[1]
        cx = min(w, 640)
        cy = min(h, 480)
        padimage = np.zeros(outsize, dtype=np.uint8)
        padimage[:cy, :cx] = image[:cy, :cx]
        return padimage, outsize[0], outsize[1]

    def inverdepth(self, param, K_inv, xy1map):
        n_d = param[:3] / np.clip(param[3], 1e-8, 1e8)  # meter n*1/d
        n_d = np.transpose(n_d, [1, 2, 0])
        inverdepth = -1 * np.sum(np.dot(n_d, K_inv) * xy1map, axis=2)
        return inverdepth

    def convert_suntos3d(self):
        adr = f'data/SUNRGBD/sunrgbd_{self.phase}.json'
        data = json.load(open(adr))
        imgs = data['images']
        annos = data['annotations']
        imgid2anno = defaultdict(list)
        id2imgid = defaultdict(list)
        for i in range(len(annos)):
            imgid2anno[annos[i]['image_id']].append(annos[i])
        for i in range(len(imgs)):
            id2imgid[i] = imgs[i]['id']

        annotations = []
        for i in range(len(imgs)):
            im_name = imgs[i]['file_name'][6:]
            im_name = os.path.join('data', 'SUNRGBD', im_name)
            im = cv2.imread(im_name)
            h, w, _ = im.shape
            anno = imgid2anno[id2imgid[i]]
            sample = {}
            sample['file_name'] = im_name
            sample['layout'] = []
            for j, an in enumerate(anno):
                # seg = np.array(an['segmentation']).reshape(-1, 2)
                seg = [np.array(x).reshape(-1, 2) for x in an['segmentation']]
                line = np.array(an['inter_line']).reshape(-1,
                                                          2).astype(np.int32)
                param = np.array(an['plane_param'])
                category_id = an['category_id']
                if np.all(line == 0) or j == 0:  # no line first wall or ceiling or floor
                    sample['layout'].append({
                        'attr': 0,
                        'endpoints': [],
                        'polygon': [x.tolist() for x in seg],
                        'plane_param': param.tolist(),
                        'category': category_id
                    })
                    continue
                if len(line) == 4:  # oc line
                    left = line[0:2]
                    right = line[2:]
                    len_l = np.sum((left[0] - left[1])**2)
                    len_r = np.sum((right[0] - right[1])**2)
                    if len_l < len_r:
                        endpoints = right if right[0,
                                                   1] < right[1, 1] else right[[1, 0]]
                    else:
                        endpoints = left if left[0,
                                                 1] < left[1, 1] else left[[1, 0]]
                    sample['layout'].append({
                        'attr': 1,
                        'endpoints': endpoints.tolist(),
                        'polygon': [x.tolist() for x in seg],
                        'plane_param': param.tolist(),
                        'category': category_id
                    })
                    continue
                line = line if line[0, 1] < line[1, 1] else line[[1, 0]]
                sample['layout'].append({
                    'attr': 2,
                    'endpoints': line.tolist(),
                    'polygon': [x.tolist() for x in seg],
                    'plane_param': param.tolist(),
                    'category': category_id
                })
            annotations.append(sample)

        return annotations

