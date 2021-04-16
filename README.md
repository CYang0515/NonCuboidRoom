# NonCuboidRoom

## Paper

**Learning to Reconstruct 3D Non-Cuboid Room Layout from a Single RGB Image**

[Cheng Yang](https://github.com/CYang0515/)\*,
[Jia Zheng](http://bertjiazheng.github.io/)\*,
[Xili Dai](https://github.com/Delay-Xili),
[Rui Tang](https://www.linkedin.com/in/rui-tang-50973488?originalSubdomain=cn),
[Yi Ma](https://people.eecs.berkeley.edu/~yima/),
[Xiaojun Yuan](https://faculty.uestc.edu.cn/yuanxiaojun/en/index.htm).

[[Preprint]()]
[[Supplementary Material](https://drive.google.com/file/d/1J9CC8ofxt03r2s4w5bGDZLlSL7Ke8oeK/view?usp=sharing)]

(\*: Equal contribution)

## Installation

The code is tested with Ubuntu 16.04, PyTorch v1.5, CUDA 10.1 and cuDNN v7.6.

``` bash
# create conda env
conda create -n layout python=3.6
# activate conda env
conda activate layout
# install pytorch
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
# install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Structured3D Dataset

Please download [Structured3D dataset](https://structured3d-dataset.org/) and our processed [2D line annotations](https://drive.google.com/file/d/1b-BLlXDc323WPb0bS8Jx1dm9xX80q41d/view?usp=sharing). The directory structure should look like:

``` 
data
└── Structured3D
    │── Structured3D
    │   ├── scene_00000
    │   ├── scene_00001
    │   ├── scene_00002
    │   └── ...
    └── line_annotations.json
```

### SUN RGB-D Dataset

Please download [SUN RGB-D dataset](https://rgbd.cs.princeton.edu/), our processed [2D line annotation](https://drive.google.com/drive/folders/1mZlHSrAWALytuKUUsw_E_YmBW4OZgZS3?usp=sharing) for SUN RGB-D dataset, and [layout annotations of NYUv2 303 dataset](https://cs.stanford.edu/people/zjian/project/ICCV13DepthLayout/ICCV13DepthLayout.html). The directory structure should look like:

``` 
data
└── SUNRGBD
    │── SUNRGBD
    │    ├── kv1
    │    ├── kv2
    │    ├── realsense
    │    └── xtion
    │── sunrgbd_train.json      // our extracted 2D line annotations of SUN RGB-D train set
    │── sunrgbd_test.json       // our extracted 2D line annotations of SUN RGB-D test set
    └── nyu303_layout_test.npz  // 2D ground truth layout annotations provided by NYUv2 303 dataset
```

### Pre-trained Models

You can download our pre-trained models here:

* [The model](https://drive.google.com/file/d/1DZnnOUMh6llVwhBvb-yo9ENVmN4o42x8/view?usp=sharing) trained on Structured3D dataset.
* [The model](https://drive.google.com/file/d/1ioy5xaNvxNG5GLS8gOCfCRvSvI4NydBC/view?usp=sharing) trained on SUN RGB-D dataset and NYUv2 303 dataset.

## Structured3D Dataset

To train the model on the Structured3D dataset, run this command:

``` bash
python train.py --model_name s3d --data Structured3D
```

To evaluate the model on the Structured3D dataset, run this command:

``` bash
python test.py --pretrained DIR --data Structured3D
```

## NYUv2 303 Dataset

To train the model on the SUN RGB-D dataset and NYUv2 303 dataset, run this command:

``` bash
# first fine-tune the model on the SUN RGB-D dataset
python train.py --model_name sunrgbd --data SUNRGBD --pretrained Structure3D_DIR --split all --lr_step []
# Then fine-tune the model on the NYUv2 subset
python train.py --model_name nyu --data SUNRGBD --pretrained SUNRGBD_DIR --split nyu --lr_step [] --epochs 10
```

To evaluate the model on the NYUv2 303 dataset, run this command:

``` bash
python test.py --pretrained DIR --data NYU303
```

## Inference on the customized data

To predict the results of customized images, run this command:

``` bash
python test.py --pretrained DIR --data CUSTOM
```

## Citation

``` bibtex
@article{NonCuboidRoom,
  title   = {Learning to Reconstruct 3D Non-Cuboid Room Layout from a Single RGB Image},
  author  = {Cheng Yang and
             Jia Zheng and
             Xili Dai and
             Rui Tang and
             Yi Ma and
             Xiaojun Yuan},
  journal = {CoRR},
  year    = {2021}
}
```

## LICENSE

The code is released under the [MIT license](LICENSE). Portions of the code are borrowed from [HRNet-Object-Detection](https://github.com/HRNet/HRNet-Object-Detection) and [CenterNet](https://github.com/xingyizhou/CenterNet).
