import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torchvision.transforms as tf
from PIL import Image
from torch.utils import data


class NYU303(data.Dataset):
    def __init__(self, config, phase='test', cato='nyu', exam=False):
        self.config = config
        self.phase = phase
        assert phase == 'test'
        self.cato = cato
        self.max_objs = config.max_objs
        self.exam = exam
        if exam:
            self.img_adr = 'example/'
        else:
            self.img_adr = 'data/SUNRGBD/SUNRGBD/kv1/NYUdata'
        self.anno_adr = 'data/SUNRGBD/nyu303_layout_test.npz'
        self.K = np.array([518.857901, 0.000000, 284.582449,
                           0.000000, 519.469611, 208.736166,
                           0.000000, 0.000000, 1.000000]).reshape([3, 3]).astype(np.float32)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)
        self.fullres_K = np.array([518.857901, 0.000000, 325.582449, 0.000000, 519.469611, 253.736166,
                                   0.000000, 0.000000, 1.000000]).reshape([3, 3]).astype(np.float32)  # full resolution K
        self.fullres_K_inv = np.linalg.inv(self.K).astype(np.float32)
        self.colorjitter = tf.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        self.transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        annotation = np.load(self.anno_adr)
        self.ids = annotation['index']
        self.layout = annotation['layout']

    def __getitem__(self, index):
        if self.exam:
            index = np.where(self.ids==969)[0][0]  # we provide one case to debug
        sample = self.ids[index]
        crop_img = os.path.join(
            self.img_adr, 'NYU'+str(sample).zfill(4), 'image', 'NYU'+str(sample).zfill(4)+'.jpg')
        fullres_img = os.path.join(
            self.img_adr, 'NYU'+str(sample).zfill(4), 'fullres', 'NYU'+str(sample).zfill(4)+'.jpg')
        img = Image.open(crop_img)  # RGB
        full_img = Image.open(fullres_img)
        if self.phase == 'train' and self.config.colorjitter:
            img = self.colorjitter(img)
        img = np.array(img)
        img, inh, inw = self.padimage(img)
        img = self.transforms(img)

        full_img = np.array(full_img)
        full_img = self.transforms(full_img)

        segs = self.layout[:, :, index]

        # plane detection gt and instance plane params  #center map, wh, offset, instance param
        oh, ow = inh // self.config.downsample, inw // self.config.downsample
        hm = np.zeros((3, oh, ow), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        params3d = np.zeros((self.max_objs, 4), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        ret = {'img': img, 'plane_hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'plane_wh': wh,
               'plane_offset': reg,
               'params3d': params3d}
        ret['fullimg'] = full_img
        # line detection gt  # line map, alpha, offset map
        line_hm = np.zeros((3, oh, ow), dtype=np.float32)
        ret['line_hm'] = line_hm[0:1]
        ret['line_alpha'] = line_hm[1:2]
        ret['line_offset'] = line_hm[2:3]

        # plane param map gt  # pixel-wise plane param
        plane_params = np.zeros((4, oh, ow), dtype=np.float32)
        plane_params_input = np.zeros((4, inh, inw), dtype=np.float32)
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
        inverdepth = np.zeros([oh, ow], dtype=np.float32)
        ret['odepth'] = inverdepth
        oseg = cv2.resize(segs, (ow, oh), interpolation=cv2.INTER_NEAREST)
        ret['oseg'] = oseg
        ret['oxy1map'] = oxy1map

        # reconstructure  # camera intri, plane label, segs, depth
        ret['intri'] = self.K  # np.array(anno['intri']).T
        ret['intri_inv'] = self.K_inv
        ret['full_intri'] = self.fullres_K
        ret['full_intri_inv'] = self.fullres_K_inv

        # evaluate gt
        ret['ilbox'] = np.zeros((20,), dtype=np.float32)
        ret['iseg'] = segs
        ixymap = cv2.resize(xymap, (inw, inh), interpolation=cv2.INTER_LINEAR)
        ixy1map = np.concatenate([ixymap, np.ones_like(
            ixymap[:, :, :1])], axis=-1).astype(np.float32)
        inverdepth_input = np.zeros([inh, inw], dtype=np.float32)
        ret['ixy1map'] = ixy1map
        ret['idepth'] = inverdepth_input

        return ret

    def __len__(self):
        return len(self.ids)

    def padimage(self, image):
        outsize = [480, 640, 3]
        h, w = image.shape[0], image.shape[1]
        cx = min(w, 640)
        cy = min(h, 480)
        padimage = np.zeros(outsize, dtype=np.uint8)
        padimage[:cy, :cx] = image[:cy, :cx]
        return padimage, outsize[0], outsize[1]