# Good helper is around you: Attention-driven Masked Image Modeling

This repository contains PyTorch implementation of AMT(Attention-driven masking and throwing strategy) with **MAE** and **SimMIM**.  
For details see [Good helper is around you: Attention-driven Masked Image Modeling](http://arxiv.org/abs/2211.15362).
![AMT](./AMT.pdf)

# Preparation 
## Requirements
* Pytorch >= 1.11.0
* Python >= 3.8
* timm == 0.5.4

To configure environment, you can run:
```
conda create -n AMT python=3.8 -y
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```
## Points for Attention 
* We slightly modify the *vision_transfomer.py* in *timm.models* to provide attention map during pretraining:
```python 
class Attention(nn.Module):
...
    return x,attn # add attn 
...
```
```python
class Block(nn.Module):
...
    def forward(self, x, return_attention=False):# add return_attention as a flag
        y,attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        ...
...
```
* We only support ViT-B/16 now.



## Datasets
Please download and organize the datasets in this structure:
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
      ...
    class2/
      img2.jpeg
      ...
  val/
    class1/
      img3.jpeg
      ...
    class/2
      img4.jpeg
      ...
```




# Getting Started
This repo supports both attention-driven masking and AMT strategies, you can switch by setting mask_ratio and throw_ratio.


We test on a 4-gpu server, accum_iter is for mataining effective batch size with the original method. The following script is an example of AMT with MAE.

## Pre-training with MAE
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
    --model vit_base_patch16 \
    --batch_size 128 \
    --accum_iter 8 \
    --blr 1.5e-4 \
    --data_path ${IMAGENET_DIR}
```

## Finetuning with MAE

```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --accum_iter 8 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

## Linprobing with MAE

```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_linprobe.py \
    --accum_iter 8 \
    --batch_size 512 \
    --model vit_base_patch16 --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

This code is written by Zhengqi Liu.

ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Jie Gui (guijie@ustc.edu).
