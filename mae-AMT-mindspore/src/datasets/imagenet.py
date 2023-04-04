# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""create train or eval dataset."""

import os
import json

from PIL import Image

import mindspore
import mindspore.dataset as de
import mindspore.ops as ops
from mindspore.dataset.vision import Inter

import mindspore.dataset.vision.c_transforms as C
import mindspore._c_dataengine as cde




class MyImageFolder(de.ImageFolderDataset):
    def __init__(self, root, **kwargs):

        super().__init__(root = root, **kwargs)

        self.mask_weights = mindspore.numpy.tile(ops.Ones(1, 14, 14), (self.__len__(),1,1))


    def __getitem__(self, index: int):

        path, target = self.samples[index]
        sample = self.loader(path)
        mask_weights = self.mask_weights[index]
        if self.transform is not None:
            sample, mask_weights = self.transform(sample,mask_weights)

        return sample, target, mask_weights


    
class DataLoader:
    def __init__(self, imgs_path, data_dir=None):
        """Loading image files as a dataset generator."""
        imgs_path = os.path.join(data_dir, imgs_path)
        assert os.path.realpath(imgs_path), "imgs_path should be real path."
        with open(imgs_path, 'r') as f:
            data = json.load(f)
        if data_dir is not None:
            data = [os.path.join(data_dir, item) for item in data]
        self.data = data
        self.mask_weights = mindspore.numpy.tile(ops.Ones(196), (self.__len__()))

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        mask_weights = self.mask_weights[index]
        return (img, mask_weights)

    def __len__(self):
        return len(self.data)






def create_dataset(args):
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    dataset = de.GeneratorDataset(
        source=DataLoader(args.img_ids, data_dir=args.data_path),
        column_names="image", num_shards=args.device_num,
        shard_id=args.local_rank, shuffle=True)

    dataset_val = de.ImageFolderDataset(dataset_dir=args.data_path, decode=True, num_shards=args.device_num, shard_id=args.local_rank, shuffle=False)


    transform_eval = [
            C.Resize((224, 224)),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
            ]
        
    trans = [
        C.RandomResizedCrop(
            args.image_size,
            scale=(0.2, 1.0),
            ratio=(0.75, 1.333),
            interpolation=Inter.BICUBIC
        ),
        C.RandomHorizontalFlip(prob=0.5),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW(),
    ]

    ds = dataset.map(num_parallel_workers=args.num_workers,
                     operations=trans, python_multiprocessing=True)
    ds = ds.shuffle(buffer_size=10)
    ds = ds.batch(args.batch_size, drop_remainder=True)

    ds = ds.repeat(1)

    ds_val = dataset_val.map(input_columns="image", num_parallel_workers=args.num_workers,
                     operations=transform_eval, python_multiprocessing=True)
    ds_val = ds_val.shuffle(buffer_size=10)
    ds_val = ds_val.batch(args.batch_size, drop_remainder=False)

    ds_val = ds_val.repeat(1)



    return ds, ds_val
