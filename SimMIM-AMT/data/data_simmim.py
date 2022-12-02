# --------------------------------------------------------
# References:
# SimMIM: https://github.com/microsoft/SimMIM
# --------------------------------------------------------

import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
# import torchvision
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from . import amtdatasets as ds



def build_loader_simmim(config, logger):

    transform_train = ds.trainCompose([
            # T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            ds.maskRandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            ds.maskRandomHorizontalFlip(),
            ds.maskgenerate(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
    # transform = SimMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform_train}')

    dataset = ds.MyImageFolder(config.DATA.DATA_PATH, transform = transform_train)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    
    sampler = DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True)
    

    #dataset_eval
    transform_eval = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_eval = ImageFolder(config.DATA.DATA_PATH,transform = transform_eval)
    logger.info(f'Build dataset: eval images = {len(dataset)}')
    sampler_eval = ds.SequentialDistributedSampler(
            dataset_eval,config.DATA.BATCH_SIZE_UP, num_replicas=num_tasks, rank=global_rank
        )
    logger.info("Sampler_eval = %s" % str(sampler_eval))

    data_loader_eval = DataLoader(
        dataset_eval, sampler = sampler_eval,
        batch_size = config.DATA.BATCH_SIZE_UP,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    len_ds = len(dataset_eval)

    return dataset,dataset_eval,dataloader, data_loader_eval, len_ds