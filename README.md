This repo is

(1) a PyTorch library that provides classical knowledge distillation algorithms on image retrieval benchmarks.

(2) the official implementation of the Pattern Recognition-2024 paper: [Pairwise difference relational distillation for object re-identification](https://www.sciencedirect.com/science/article/pii/S0031320324002061).



## Pairwise difference relational distillation for object re-identification

### Framework
<div style="text-align:center"><img src="/AIR_Distiller/.github/D3still_framework.png" width="100%" ></div>


## SOTA Experiments

### On the Market-1501 dataset
| Teacher <br> Student | ResNet101 ($256\times256$) <br> ResNet18 ($256\times256$)|ResNet101 ($320\times160$) <br> ResNet18 ($320\times160$)|
|:---------------:|:-----------------:|:---------------:|
| VanillaKD | 81.54% mAP &nbsp; 92.61% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| FitNet | 78.97% mAP &nbsp; 91.57% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| AT | 80.69% mAP &nbsp; 92.67% R1| 0.80% mAP &nbsp; 0.28% R1 |
| CC | 77.03% mAP &nbsp; 90.35% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| SP | 80.28% mAP &nbsp; 92.40% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| RKD | 81.59% mAP &nbsp; 92.34% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| PKT | 79.73% mAP &nbsp; 91.75% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| CSD | 79.09% mAP &nbsp; 91.72% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| D3 | 81.57% mAP &nbsp; 92.52% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| PDRD (Ours) | **83.49%** mAP &nbsp; **93.47%** R1 | 0.80% mAP &nbsp; 0.28% R1 |

### On the DukeMTMC_reID dataset
| Teacher <br> Student | ResNet101 ($256\times256$) <br> ResNet18 ($256\times256$)|ResNet101 ($320\times160$) <br> ResNet18 ($320\times160$)|
|:---------------:|:-----------------:|:---------------:|
| VanillaKD | 71.77% mAP &nbsp; 85.95% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| FitNet | 68.63% mAP &nbsp; 82.85% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| AT | 71.01% mAP &nbsp; 84.69% R1| 0.80% mAP &nbsp; 0.28% R1 |
| CC | 67.42% mAP &nbsp; 82.32% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| SP | 71.45% mAP &nbsp; 84.20% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| RKD | 71.93% mAP &nbsp; 85.55% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| PKT | 70.66% mAP &nbsp; 84.38% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| CSD | 68.24% mAP &nbsp; 83.26% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| D3 | 71.27% mAP &nbsp; 84.65% R1 | 0.80% mAP &nbsp; 0.28% R1 |
| PDRD (Ours) | **73.76%** mAP &nbsp; **87.03%** R1 | 0.80% mAP &nbsp; 0.28% R1 |

# IR-Distiller

### Introduction
IR-Distiller supports the following distillation methods on Market-1501, DukeMTMC_reID, MSMT17, In-Shop Clothes Retrieval (In-Shop) and  Stanford Online Products (SOP):
|Method|Publication|YEAR|
|:---:|:---:|:---:|
|[VanillaKD](https://arxiv.org/abs/1503.02531) |NIPS Workshop|2014|
|[FitNet](https://arxiv.org/abs/1412.6550) |ICLR|2015 |
|[AT](https://openreview.net/pdf?id=Sks9_ajex) |ICLR|2017 |
|[CC](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html) |ICCV| 2019|
|[SP](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf) |ICCV| 2019|
|[PKT](https://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf) | ECCV | 2018 |
|[RKD](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) |CVPR| 2019|
|[CSD](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.html) |CVPR|2023 |
|[D3still](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html) |CVPR|2024|
|[PDRD](https://www.sciencedirect.com/science/article/pii/S0031320324002061)|PR|2024|

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
- The dataset has been prepared in the format we read at the link: https://pan.baidu.com/s/1Eh4ob9IdtJ42_0MUjEPE8g?pwd=wnvx or [https://drive.google.com/drive/folders/1OBOHWP15sH2mdgaKhq6v3YImT9zd__gj?usp=drive_link](https://drive.google.com/drive/folders/1tcb6GR8yLwTxatK8vDIvcSnYzc_KoIPW?usp=sharing). Please download the data and untar it to `XXXX/data` via `unzip XXXX`. For example,  `unzip CUB_200_2011.zip`. Finally, the data file directory should be as follows:


  XXXX/data/  
    &nbsp; &nbsp; &nbsp; &nbsp; └── Market1501
    &nbsp; &nbsp; &nbsp; &nbsp; └── DukeMTMC_reID
    &nbsp; &nbsp; &nbsp; &nbsp; └── MSMT17
    &nbsp; &nbsp; &nbsp; &nbsp; └── VeRi776
    &nbsp; &nbsp; &nbsp; &nbsp; └── InShop  
    &nbsp; &nbsp; &nbsp; &nbsp; └── Stanford_Online_Products  


1. download teacher models
- Our teacher models are at https://pan.baidu.com/s/1X8urI8_bDfmdapSaNGYbtA?pwd=if2i or https://drive.google.com/drive/folders/1-S6r2nrcn6fQzBrnnEtLbivs4sZ028ZE?usp=drive_link, please download the checkpoints to `./download_ckpts`

2. Path setting
- Please modify the following line in `IR_Distiller/tools/train.py` and `IR_Distiller/tools/test.py`:  
`sys.path.append(os.path.abspath("XXXXX/IR_Distiller"))`  
Replace `"XXXXX/IR_Distiller"` with the absolute path of your project to ensure correct module imports.

 **Example** (assuming the project path is `/home/user/IR_Distiller`):  
```python
import sys  
import os  
sys.path.append(os.path.abspath("/home/user/IR_Distiller"))
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
  # for instance, when the teacher network is ResNet101 and the student network is ResNet18, our PDRD method.
  python IR_Distiller/tools/train.py --cfg Training_Configs/SOP/ResNet101_ResNet18_256x256/PDRD.yaml 
  ```

  - By default, the ImageNet pre-trained model will be used for training. The model will be automatically downloaded from the internet on the first run.  
  If you want to use a different pre-trained model, modify the `STUDENT_PRETRAIN_PATH` in the YAML configuration file.  


4. Evaluation

 ```bash
  # for instance, when the teacher network is ResNet101 and the student network is ResNet18, our PDRD method.
  python AIR_Distiller/tools/test.py --cfg Training_Configs/SOP/ResNet101_256x256_ResNet18/PDRD.yaml 
 ```


 - During inference, you can first navigate to `IR_Distiller/utils/rank_cylib` and run the following commands to enable sorting with C language, which helps reduce inference time:  

```bash
python3 setup.py build_ext --inplace
rm -rf build
```

### Custom Distillation Method

1. create a python file at `IR_Distiller/distillers/` and define the distiller
  
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

2. regist the distiller in `distiller_dict` at `IR_Distiller/distillers/__init__.py`

3. regist the corresponding hyper-parameters at `IR_Distiller/config/defaults.py`

4. create a new config file and test it.

### Experimental Note
During training, the batch size of the distillation dataloader (96) is equal to the student dataloader (96). However, some method-specific hyperparameters may need to be tuned accordingly.

# Citation

If this repo is helpful for your research, please consider citing these paper:

```BibTeX
@article{xie2024pairwise,
  title={Pairwise difference relational distillation for object re-identification},
  author={Xie, Yi and Wu, Hanxiao and Lin, Yihong and Zhu, Jianqing and Zeng, Huanqiang},
  journal={Pattern Recognition},
  volume={152},
  pages={110455},
  year={2024},
  publisher={Elsevier}
}
```

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

IR_Distiller is released under the MIT license. See [LICENSE](LICENSE) for details.

# Acknowledgement
- Thanks for DKD. We build this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller).
- Thanks for D3still. We build this library based on the [D3still's codebase](https://github.com/SCY-X/D3still).
- Thanks Yihong Lin for the discussion about PDRD.
