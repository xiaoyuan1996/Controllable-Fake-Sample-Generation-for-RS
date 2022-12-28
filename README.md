



## The offical PyTorch code for paper ["Efficient and Controllable Remote Sensing Fake Sample Generation Based on Diffusion Model.]()

##### Author: Chongyang Hao

<a href="https://github.com/xiaoyuan1996/Controllable-Fake-Sample-Generation-for-RS"><img src="https://travis-ci.org/Cadene/block.bootstrap.pytorch.svg?branch=master"/></a>
![Supported Python versions](https://img.shields.io/badge/python-3.7-blue.svg)
![Supported OS](https://img.shields.io/badge/Supported%20OS-Linux-yellow.svg)
![npm License](https://img.shields.io/npm/l/mithril.svg)
<a href="https://pypi.org/project/mitype/"><img src="https://img.shields.io/pypi/v/mitype.svg"></a>



### -------------------------------------------------------------------------------------

### Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then we'll let you know when we update

```bash
#### News:
#### 2022.12.20: ---->The code of CFSG-RS is expected to be released before next year<----
#### 2021.12.29: ---->The code of CFSG-RS has been open to access<----
#### 2021.12.30: ---->Updated usage methods and some test results<----
```

### -------------------------------------------------------------------------------------

## INTRODUCTION

This is CFSG-RS,a fake sample generation method for remote sensing images.
Here, you can get an efficient remote sensing fake sample generation  framework based on the diffusion model, which can be further modified to achieve more controllable and better generation effect.

![arch image](./figures/title.jpg)

##

## [CFSG-RS](Controllable-Fake-Sample-Generation-for-RS/README.md)

### Network Architecture

![arch image](./figures/framework-RS.jpg)
The proposed training framework for efficient and controllable remote sensing fake sample generation.
  (a) The original reverse diffusion process.
  (b) The designed diffusion distillation based on multi-frequency knowledge transfer.
  (c) Progressive training strategy for accelerated diffusion learning.

### Multi-frequency Dynamic Knowledge Distillation.

### Progressive Training Strategy

![arch image](./figures/freq.png)

The diffusion model learns too slowly, and the general network starts from low frequency and learns high frequency. For this reason, we optimize the convolutional kernels at different scales by staging the diffusion model to learn different discriminative solutions.

![arch image](./figures/compare_strategy.jpg)

## Citation

If you feel this code helpful or use this code or dataset, please cite it as

```

```

## Usage

### Environment

```python
pip install -r requirement.txt
```

### Pretrained Model

We prepared three Pretrained Models, representing the regular model, the unguided lightweight model, and the tuned lightweight model. The resource consumption of the light-weighted model is much smaller than that of the regular model. The original model contains 97,807,491 parameters, while the optimized diffusion model requires only 42,054,851 parameters to complete the pseudo-sample generation task.

| Type                                                        | Platform（Code：qwer) |
| ----------------------------------------------------------- | --------------------- |
| nomal_model                                                 |                       |
| unlead_light_model                                          |                       |
| leader_light_model                                          |                       |
| inception_v3_google-0cc3c7bd.pth(use when eval FID and IS ) |                       |

```python
# Download the pretrain model and edit [false_generate]_[leader_network].json about "resume_state":
"resume_state": [your pretrain model path]
```

### Data Prepare

#### New Start

If you didn't have the data, you can prepare it by following steps:

train dataset

testset

then you need to change the datasets config to your data path and image resolution: 

```json
"datasets": {
        "train": {
            "name": "test_process",
            "mode": "HR", // whether need LR img
            "dataroot": "/data/diffusion_data/dataset/false_generate",//train dastset root path
            "datatype": "random", //lmdb or img, path of img files
            "l_resolution": 32,
            "r_resolution": 256, // crop size
            "batch_size": 2,
            "num_workers": 2,
            "use_shuffle": true,
            "data_len": -1 // -1 represents all data used in train
        },
        "val": {
            "name": "test_process",
            "mode": "HR",
            "dataroot": "/data/diffusion_data/val/test",//path of img files
            "datatype": "infer", //infer,random,crop，multiple
            "l_resolution": 32,
            "r_resolution": 256,
            "data_len": -1 // data length in validation
        }
    },
```

```shell
 #Training data file structure
/data/diffusion_data/dataset/false_generate
├── hr_256 # ground-truth images.
└── sr_32_256 # mask RGB images.
```

```python
 #Test data file structure
/data/diffusion_data/val/test
├── images # ground-truth images.
└── labels # mask RGB images.
```

You can also adjust your file structure by modifying the data/LRHR_dataset.py file

```python
self.sr_path = Util.get_paths_from_images(
                '{}/labels'.format(dataroot))
self.hr_path = Util.get_paths_from_images(
                '{}/images'.format(dataroot))
```

### Training/Resume Training

The Normal Training

```python
# Use the python file in the model_train folder to complete the routine training 
python train_sr.py -p train -c config/false_generate.json
```

The Light_network Leader Training

```python
# Use the python file in the light--model folder to complete the training, the bootstrap training fixed bootstrap model

python leader_sr.py -p train -c config/leader_network.json

#Determine the network structure by changing the which_model_G property in the json file
"model": {
        "which_model_G属性": "sr3", // use the feature or sr3 network structure
}
```

```python
# Use the python file in the model_leader folder to complete the training, which trains both the bootstrap model and the lightweight model

python fusion_sr.py -p train -c config/new_leader.json

```

### Test And Infer

```python
# Quantitative evaluation alone using FID/BRISQUE metrics on given result root
python train_infer.py -c [config file] -i
```

### 存储路径

You can change the weight file and output image storage location under the file core/logger.py

```python
    if args.infer:
        experiments_root = os.path.join(
            '/data/diffusion_data/infer', '{}_{}'.format(opt['name'], get_timestamp()))
    else:
        experiments_root = os.path.join(
            '/data/diffusion_data/experiments', '{}_{}'.format(opt['name'], get_timestamp()))
    opt['path']['experiments_root'] = experiments_root
```

### Results

Samples of Inference Process

| <img src="./misc/0_32_sr_process.png" alt="show" style="zoom:90%;" /> | <img src="./misc/0_64_sr_process.png" alt="show" style="zoom:90%;" /> | <img src="./misc/0_111_sr_process.png" alt="show" style="zoom:90%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./misc/0_142_sr_process.png" alt="show" style="zoom:90%;" /> | <img src="./misc/0_425_sr_process.png" alt="show" style="zoom:90%;" /> | <img src="./misc/0_435_sr_process.png" alt="show" style="zoom:90%;" /> |

- #### # Controllable-Fake-Sample-Generation-for-RS
