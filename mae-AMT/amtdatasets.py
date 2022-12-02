import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

import math


class MyImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, **kwargs):
        
        super().__init__(root = root, **kwargs)
        
        #init weights for sampling in AMT
        self.mask_weights = torch.ones(1, 14, 14).repeat(self.__len__(),1,1)


    def __getitem__(self, index: int):

        path, target = self.samples[index]
        sample = self.loader(path)
        mask_weights = self.mask_weights[index]
        if self.transform is not None:
            sample, mask_weights = self.transform(sample,mask_weights)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, mask_weights


class trainCompose(transforms.Compose):
    def __call__(self, img, mask): 

        #mask needs to be processed individually for some operations
        for i in self.transforms[:2]:
            img, mask = i(img,mask)

        for t in self.transforms[2:]:
            img = t(img)

        return img, mask


class maskRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)
    
    def forward(self, img, mask):

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        mask = mask[i//img.size[1] : (i+h)//img.size[1] + 2 , j//img.size[0] : (j+w)//img.size[0] + 2]


        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), F.resize(mask.unsqueeze(0), self.size).clamp(min=0,max=1)

class maskRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self):
        super().__init__()

    def forward(self, img, mask):

        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(mask)
        return img, mask



#From https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
 
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas
 
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)
 
    def __len__(self):
        return self.num_samples
