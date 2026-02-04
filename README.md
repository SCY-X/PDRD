This repo is

(1) a PyTorch library that provides classical knowledge distillation algorithms on image retrieval benchmarks.

(2) the official implementation of the Pattern Recognition-2024 paper: [Pairwise difference relational distillation for object re-identification](https://www.sciencedirect.com/science/article/pii/S0031320324002061).




NOTE: 1) Unlike our CVPR 2024 paper, which employs only distillation loss, this repository follows common practices in prior knowledge distillation research by incorporating standard losses (e.g., cross-entropy loss and triplet loss) during the distillation of the student network (query network). Correspondingly, the hyperparameters in all distillation methods are also adjusted accordingly. As a result, the experimental outcomes reported in this repository demonstrate significant performance improvements across multiple datasets compared to those presented in the CVPR 2024 paper. For example, on the In-Shop dataset, [FitNet](https://arxiv.org/abs/1412.6550) performance is improved from 62.84% mAP to 65.99% mAP. To offer a more comprehensive evaluation of the effectiveness of our approach, this repository presents ablation experiment results on new benchmarks. 2) The mINP metric is only applicable and meaningful for the MSMT17 dataset.

(3) the official implementation of the Neural Networks-2025 paper: [Unambiguous granularity distillation for asymmetric image retrieval](https://www.sciencedirect.com/science/article/pii/S0893608025001820).

## What's New
## Jan 23, 2026
Fixed an issue where the downsampling stride in the final stage of ResNet-IBN was not set to 1. Based on the corrected architecture, we re-trained the ResNet101-IBN model and released the corresponding weights on both Baidu Cloud and Google Drive under the filename MSMT17_ResNet101_IBN_320x160_65.15_85.46.pth. In addition, we updated the distillation results on the MSMT17 dataset for ResNet101-IBN → ResNet18.

## D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval

### Framework
<div style="text-align:center"><img src="/AIR_Distiller/.github/D3still_framework.png" width="100%" ></div>

### Ablation Experiments

Gallery Network: ResNet101 &nbsp; Gallery Network Input Resolution: $256\times256$
 
Query Network: ResNet18  &nbsp; Query Network Input Resolution: CUB-200-2011 ($128\times128$) &nbsp; In-Shop ($64\times 64$) &nbsp; SOP ($64\times 64$)

<div style="text-align:center"><img src="/AIR_Distiller/.github/D3still_ablation_study.png" width="100%" ></div> 

## Unambiguous granularity distillation for asymmetric image retrieval

### Framework
<div style="text-align:center"><img src="/AIR_Distiller/.github/UGD_framework.png" width="100%" ></div> 

## SOTA Experiments

### On the Caltech-UCSD Birds 200 (CUB-200-2011) dataset

|




# IR-Distiller

### Introduction

AR-Distiller supports the following distillation methods on In-Shop Clothes Retrieval (In-Shop), Stanford Online Products (SOP) and MSMT17:
|Method|Publication|YEAR|
|:---:|:---:|:---:|
|[VanillaKD](https://arxiv.org/abs/1503.02531) |NIPS Workshop|2014|
|[FitNet](https://arxiv.org/abs/1412.6550) |ICLR|2015 |
|[PKT](https://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf) | ECCV | 2018 |
|[RKD](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) |CVPR| 2019|
|[CC](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html) |ICCV| 2019|
|[CSD](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.html) |CVPR|2023 |
|[RAML](https://openaccess.thecvf.com/content/WACV2023/html/Suma_Large-to-Small_Image_Resolution_Asymmetry_in_Deep_Metric_Learning_WACV_2023_paper.html)|WACV|2023|
|[ROP](https://openreview.net/forum?id=dYHYXZ3uGdQ)|ICLR|2023|
|[D3still](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html) |CVPR|2024|
|[UGD](https://www.sciencedirect.com/science/article/pii/S0893608025001820) |Neural Networks|2025|

### Installation

Environments:

- Python 3.10
- PyTorch 2.4.1
- torchvision 0.19.1
- ptflops 0.7.4

Install the package:

```
sudo pip3 install -r requirements.txt
```

### Getting started

0. download data
- The dataset has been prepared in the format we read at the link: https://pan.baidu.com/s/1ySKEmn8WVm2efJVvJ_vMBQ?pwd=ebyx or https://drive.google.com/drive/folders/1OBOHWP15sH2mdgaKhq6v3YImT9zd__gj?usp=drive_link. Please download the data and untar it to `XXXX/data` via `unzip XXXX`. For example,  `unzip CUB_200_2011.zip`. Finally, the data file directory should be as follows:


  XXXX/data/  
    &nbsp; &nbsp; &nbsp; &nbsp; └── CUB_200_2011  
    &nbsp; &nbsp; &nbsp; &nbsp; └── InShop  
    &nbsp; &nbsp; &nbsp; &nbsp; └── Stanford_Online_Products  
    &nbsp; &nbsp; &nbsp; &nbsp; └── MSMT17

1. download teacher models
- Our teacher models are at https://pan.baidu.com/s/1X8urI8_bDfmdapSaNGYbtA?pwd=if2i or https://drive.google.com/drive/folders/1-S6r2nrcn6fQzBrnnEtLbivs4sZ028ZE?usp=drive_link, please download the checkpoints to `./download_ckpts`

2. Path setting
- Please modify the following line in `AIR_Distiller/tools/train.py` and `AIR_Distiller/tools/test.py`:  
`sys.path.append(os.path.abspath("XXXXX/AIR_Distiller"))`  
Replace `"XXXXX/AIR_Distiller"` with the absolute path of your project to ensure correct module imports.

 **Example** (assuming the project path is `/home/user/AIR_Distiller`):  
```python
import sys  
import os  
sys.path.append(os.path.abspath("/home/user/AIR_Distiller"))
```
- Please set the `ROOT_DIR` path in the configuration file, i.e., XXX.yaml to the absolute path of the `data` folder.  
- 
**Example** (assuming the data path is `/home/user/data`):  
```yaml
DATASETS:
  NAMES: "SOP"
  ROOT_DIR: "/home/user/data"
```


3. Training 

 ```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our D3 method.
  python AIR_Distiller/tools/train.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/D3.yaml 
  ```
 ```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our UGD method.
  python AIR_Distiller/tools/train.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/UGD.yaml 
  ```

  - By default, the ImageNet pre-trained model will be used for training. The model will be automatically downloaded from the internet on the first run.  
  If you want to use a different pre-trained model, modify the `STUDENT_PRETRAIN_PATH` in the YAML configuration file.  


4. Evaluation

 ```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our D3 method.
  python AIR_Distiller/tools/test.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/D3.yaml 
 ```

```bash
  # for instance, when the gallery network is ResNet101 and the query network is ResNet18, our UGD method.
  python AIR_Distiller/tools/test.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18_64x64/UGD.yaml 
 ```

 - During inference, you can first navigate to `AIR_Distiller/utils/rank_cylib` and run the following commands to enable sorting with C language, which helps reduce inference time:  

```bash
python3 setup.py build_ext --inplace
rm -rf build
```

### Custom Distillation Method

1. create a python file at `AIR_Distiller/distillers/` and define the distiller
  
  ```python
  from ._base import Distiller

  class MyDistiller(Distiller):
      def __init__(self, student, teacher, cfg):
          super(MyDistiller, self).__init__(student, teacher)
          self.hyper1 = cfg.MyDistiller.hyper1
          ...

      def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):
          # return the output logits and a Dict of losses
          ...
      # rewrite the get_learnable_parameters function if there are more nn modules for distillation.
      # rewrite the get_extra_parameters if you want to obtain the extra cost.
    ...
  ```

2. regist the distiller in `distiller_dict` at `AIR_Distiller/distillers/__init__.py`

3. regist the corresponding hyper-parameters at `AIR_Distiller/config/defaults.py`

4. create a new config file and test it.

### Experimental Note
During training, the batch size of the distillation dataloader (256) is larger than that of the student dataloader (96), resulting in fewer iterations per epoch. This discrepancy may lead to suboptimal performance  for some methods due to insufficient training steps. Future researchers can reduce the distillation batch size or increase the number of training epochs to address this concern.

Across all methods we experimented with, we found that using a distillation batch size of either 256 or 96 yields comparable best performance. However, some method-specific hyperparameters may need to be tuned accordingly.

# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@InProceedings{Xie_2024_CVPR,
    author    = {Xie, Yi and Lin, Yihong and Cai, Wenjie and Xu, Xuemiao and Zhang, Huaidong and Du, Yong and He, Shengfeng},
    title     = {D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {17181-17190}
}
```

```BibTeX
@article{zhang2025unambiguous,
  title={Unambiguous granularity distillation for asymmetric image retrieval},
  author={Zhang, Hongrui and Xie, Yi and Zhang, Haoquan and Xu, Cheng and Luo, Xuandi and Chen, Donglei and Xu, Xuemiao and Zhang, Huaidong and Heng, Pheng Ann and He, Shengfeng},
  journal={Neural Networks},
  volume={187},
  pages={107303},
  year={2025},
  publisher={Elsevier}
}
```

# License

AIR_Distiller is released under the MIT license. See [LICENSE](LICENSE) for details.

# Acknowledgement
- Thanks for DKD. We build this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller).
- Thanks Yihong Lin for the discussion about D3still.
