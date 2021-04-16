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


class Structured3D(data.Dataset):
    def __init__(self, config, phase='training'):
        self.config = config
        self.phase = phase
        self.max_objs = config.max_objs
        self.transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.K = np.array([[762, 0, 640], [0, -762, 360],
                           [0, 0, 1]], dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)
        self.colorjitter = tf.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5) 
        self.adr = os.path.join('data', 'Structured3D', 'line_annotations.json')
        with open(self.adr, 'r') as f:
            files = json.load(f)
        self.data_set = defaultdict(list)
        for _, i in enumerate(files):
            img_name = i[0]
            scene = int(img_name.split('_')[1])
            if scene <= 2999:
                self.data_set['training'].append(i)
            elif scene <= 3249:
                self.data_set['validation'].append(i)
            else:
                self.data_set['test'].append(i)

    def __getitem__(self, item):
        sample = self.data_set[self.phase][item]
        s0, s1, r, p = sample[0].split('_')[0:4]
        s = s0 + '_' + s1
        p = p.rstrip('.png')
        dirs = os.path.join('data', 'Structured3D/Structured3D', s, '2D_rendering', r, 'perspective/full', p)
        img_name = os.path.join(dirs, 'rgb_rawlight.png')
        layout_name = os.path.join(dirs, 'layout.json')

        img = Image.open(img_name)  # RGB
        inh, inw = self.config.input_h, self.config.input_w
        orih, oriw = img.size[1], img.size[0]
        ratio_w = oriw / inw
        ratio_h = orih / inh
        assert ratio_h == ratio_w == 2
        if self.phase == 'training' and self.config.colorjitter:
            img = self.colorjitter(img)
        img = np.array(img)[:, :, :-1]
        img = cv2.resize(img, (inw, inh), interpolation=cv2.INTER_LINEAR)
        img, inh, inw = self.padimage(img)
        img = self.transforms(img)

        pparams, labels, segs, endpoints = self.dataload(
            layout_name, sample[1], ratio_h, inh, inw)

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

        ret = {'img': img, 'plane_hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'plane_wh': wh, 'plane_offset': reg,
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
            plane_params[3, oseg == i] = param[3]  # offset
            plane_params_input[:3, segs == i] = param[:3, np.newaxis]
            plane_params_input[3, segs == i] = param[3]
        ret['plane_params'] = plane_params

        # param params for depth loss
        # coordinate map
        x = np.arange(ow * 8)
        y = np.arange(oh * 8)
        xx, yy = np.meshgrid(x, y)
        xymap = np.stack([xx, yy], axis=2).astype(np.float32)
        oxymap = cv2.resize(xymap, (ow, oh), interpolation=cv2.INTER_LINEAR)
        oxy1map = np.concatenate([oxymap, np.ones_like(
            oxymap[:, :, :1])], axis=-1).astype(np.float32)
        inverdepth = self.inverdepth(plane_params, self.K_inv, oxy1map)
        ret['odepth'] = inverdepth
        ret['oseg'] = oseg
        ret['oxy1map'] = oxy1map

        # reconstructure camera intri, plane label, segs, depth
        ret['intri'] = self.K
        ret['intri_inv'] = self.K_inv

        # evaluate gt
        ret['ilbox'] = np.concatenate(
            [np.array(labels), np.zeros(20-len(labels))], axis=0)
        ret['iseg'] = segs
        ixymap = cv2.resize(xymap, (inw, inh), interpolation=cv2.INTER_LINEAR)
        ixy1map = np.concatenate([ixymap, np.ones_like(
            ixymap[:, :, :1])], axis=-1).astype(np.float32)
        inverdepth_input = self.inverdepth(
            plane_params_input, self.K_inv, ixy1map)
        ret['ixy1map'] = ixy1map
        ret['idepth'] = inverdepth_input

        return ret

    def inverdepth(self, param, K_inv, xy1map):
        n_d = param[:3] / np.clip(param[3], 1e-8, 1e8)  # meter n*1/d
        n_d = np.transpose(n_d, [1, 2, 0])
        inverdepth = -1 * np.sum(np.dot(n_d, K_inv) * xy1map, axis=2)
        return inverdepth

    def padimage(self, image):
        outsize = [384, 640, 3]
        h, w = image.shape[0], image.shape[1]
        padimage = np.zeros(outsize, dtype=np.uint8)
        padimage[:h, :w] = image
        return padimage, outsize[0], outsize[1]

    def dataload(self, layout_name, lines, ratio_h, inh, inw):
        # planes
        with open(layout_name, 'r') as f:
            anno_layout = json.load(f)
            junctions = anno_layout['junctions']
            planes = anno_layout['planes']

            coordinates = []
            for k in junctions:
                coordinates.append(k['coordinate'])
            coordinates = np.array(coordinates) / ratio_h

            pparams = []
            labels = []
            segs = -1 * np.ones([inh, inw])
            i = 0
            for pp in planes:
                if len(pp['visible_mask']) != 0:
                    if pp['type'] == 'wall':
                        cout = coordinates[pp['visible_mask'][0]]
                        polygon = Polygon(cout)
                        if polygon.area >= 1000:
                            cout = cout.astype(np.int32)
                            cv2.fillPoly(segs, [cout], color=i)
                            pparams.append([*pp['normal'], pp['offset'] / 1000.])
                            labels.append(0)
                            i = i + 1
                    else:
                        for v in pp['visible_mask']:
                            cout = coordinates[v]
                            polygon = Polygon(cout)
                            if polygon.area > 1000:
                                cout = cout.astype(np.int32)
                                cv2.fillPoly(segs, [cout], color=i)
                                pparams.append([*pp['normal'], pp['offset'] / 1000.])
                                if pp['type'] == 'floor':
                                    labels.append(1)
                                else:
                                    labels.append(2)
                                i = i + 1
        # lines
        endpoints = []
        for line in lines:
            if line[-1] == 2:  # occlusion line
                points = np.array([*line[4], *line[5]]).reshape(2, -1) / ratio_h
                ymin = np.min(points[1])
                ymax = np.max(points[1])
                x0 = line[2] * ymin + line[3] / ratio_h
                x1 = line[2] * ymax + line[3] / ratio_h
                endpoints.append([x0, ymin, x1, ymax])  # start/end point
            elif line[-1] == 1:  # wall/wall line
                wall_id, endpoint = line[0:2], line[2:4]
                xy = coordinates[endpoint]
                if (xy[0, 1] - xy[1, 1]) == 0:
                    continue
                endpoints.append([xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1]])
            elif line[-1] == 3:  # floor/wall line
                pass
            else:  # ceiling/wall line
                pass
        return pparams, labels, segs, endpoints

    def __len__(self):
        return len(self.data_set[self.phase])
