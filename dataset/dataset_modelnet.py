import os
from os.path import join
import numpy as np
from dataset.lib.Dataset_Base import Dataset_Base
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch3d import transforms as trans
from dataset.dataloader_utils import MixDataset

cate10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
          'monitor', 'night_stand', 'sofa', 'table', 'toilet']


class ModelNetDataset(Dataset_Base):
    def __init__(
        self, data_dir, category, collection="train", net_arch="alexnet", aug=None
    ):
        super(ModelNetDataset, self).__init__(
            data_dir, category, collection, net_arch, 1.0
        )
        self.aug = aug

    def __getitem__(self, idx):
        rc = self.recs[idx]
        cate = rc.category
        img_id = rc.img_id
        quat = rc.so3.quaternion
        quat = torch.from_numpy(quat)
        rot_mat = trans.quaternion_to_matrix(quat)

        img = self._get_image(rc)
        img = torch.from_numpy(img)

        if self.aug is not None:
            img = self.aug(img)

        sample = dict(
            idx=idx,
            cate=self.cate2ind[cate],
            quat=quat,
            rot_mat=rot_mat,
            img=img,
            img_id=img_id,
        )

        return sample

def get_dataloader_modelnet(phase, config):
    if phase == "train":
        batch_size = config.batch_size
        collection = "train"
        shuffle = True
        aug = None

    elif phase == "test":
        batch_size = config.batch_size // torch.cuda.device_count()
        collection = "test"
        shuffle = False
        aug = None

    else:
        raise ValueError

    datasets = []
    for category in cate10:
        dset = ModelNetDataset(
            config.data_dir, category, collection=collection, net_arch="vgg16", aug=aug)
        datasets.append(dset)
    entire_dataset = MixDataset(datasets)
    entire_dloader = DataLoader(entire_dataset, batch_size=batch_size,
                                num_workers=config.num_workers, shuffle=shuffle, pin_memory=True)
    return entire_dloader, [DataLoader(cat_dset, batch_size=batch_size, num_workers=config.num_workers, shuffle=shuffle, pin_memory=True) for cat_dset in datasets], cate10
