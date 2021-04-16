import os

import cv2
import numpy as np
import torchvision.transforms as tf
from PIL import Image
from torch.utils import data


class CustomDataset(data.Dataset):
    def __init__(self, config, phase='test', files='example/'):
        self.config = config
        self.phase = phase
        self.max_objs = config.max_objs
        self.transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.K = np.array([[762, 0, 640], [0, -762, 360], [0, 0, 1]],
                          dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)

        self.files = files
        self.filenames = os.listdir(files)

    def padimage(self, image):
        outsize = [384, 640, 3]
        h, w = image.shape[0], image.shape[1]
        padimage = np.zeros(outsize, dtype=np.uint8)
        padimage[:h, :w] = image
        return padimage, outsize[0], outsize[1]

    def __getitem__(self, index):
        img = Image.open(self.files + self.filenames[index])
        img = img.resize((1280, 720))
        inh, inw = self.config.input_h, self.config.input_w
        orih, oriw = img.size[1], img.size[0]
        ratio_w = oriw / inw
        ratio_h = orih / inh
        assert ratio_h == ratio_w == 2
        img = np.array(img)[:, :, [0, 1, 2]]
        img = cv2.resize(img, (inw, inh), interpolation=cv2.INTER_LINEAR)
        img, inh, inw = self.padimage(img)
        img = self.transforms(img)
        ret = {'img': img}
        ret['intri'] = self.K
        ret['intri_inv'] = self.K_inv

        oh, ow = inh // self.config.downsample, inw // self.config.downsample
        x = np.arange(ow * 8)
        y = np.arange(oh * 8)
        xx, yy = np.meshgrid(x, y)
        xymap = np.stack([xx, yy], axis=2).astype(np.float32)
        oxymap = cv2.resize(xymap, (ow, oh), interpolation=cv2.INTER_LINEAR)
        oxy1map = np.concatenate([
            oxymap, np.ones_like(oxymap[:, :, :1])], axis=-1).astype(np.float32)
        ret['oxy1map'] = oxy1map

        ixymap = cv2.resize(xymap, (inw, inh), interpolation=cv2.INTER_LINEAR)
        ixy1map = np.concatenate([
            ixymap, np.ones_like(ixymap[:, :, :1])], axis=-1).astype(np.float32)
        ret['ixy1map'] = ixy1map
        ret['iseg'] = np.ones([inh, inw])
        ret['ilbox'] = np.zeros(20)
        return ret

    def __len__(self):
        return len(self.filenames)
