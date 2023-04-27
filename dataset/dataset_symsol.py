import os
import re
from pathlib import Path
from unicodedata import category
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from dataset.dataloader_utils import MixDataset

category_idx = dict(
    cone=0, cube=1, cyl=2, icosa=3,
    tet=4, sphereX=5, cylO=6, tetX=7,
)
symsol1 = ['cone', 'cube', 'cyl', 'icosa', 'tet']


class SymsolDataset(Dataset):
    def __init__(self, phase, config, category):
        self.phase = phase
        self.config = config
        self.root = Path(os.path.join(
            config.data_dir, 'symsol_dataset', phase)).expanduser()
        self.img_root = self.root / "images"
        self.category = category
        # if self.category == 'symsol1':
        # 	img_paths = []
        # 	for x in self.img_root.iterdir():
        # 		for y in symsol1:
        # 			if f'{y}_' in x.name:
        # 				img_paths.append(x)
        # 				break
        # 	self.img_paths = img_paths
        # else:
        self.img_paths = [
            x for x in self.img_root.iterdir() if f'{category}_' in x.name
        ]
        self.img_paths = sorted(self.img_paths)
        # if self.category == 'symsol1':
        # 	labels = {}
        # 	for i in symsol1:
        # 		labels[i] = np.load(self.root / "rotations.npz")[i]
        # 	self.labels = labels
        # 	self.category_size = self.labels['cone'].shape[0]
        # else:
        self.labels = np.load(self.root / "rotations.npz")[self.category]
        self.labels = torch.from_numpy(self.labels)
        assert len(self.img_paths) == self.labels.shape[0]
        self.length = len(self.img_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # if self.category == 'symsol1':
        # 	category = ''.join(re.findall(r'(.*?)_', img_path.name))
        # else:
        category = self.category
        # if self.category == 'symsol1':
        # 	index = idx % self.category_size
        # 	so3 = torch.from_numpy(self.labels[category][index])
        # else:
        so3 = self.labels[idx]
        # given ground truth of many symmetric rotations, select on at random

        # load image as np.uint8 shape (28, 28)
        so3_index = so3[0]
        x = Image.open(img_path)
        x = np.array(x)
        # convert to [0, 1.0] torch.float32, and normalize
        transform = transforms.Compose([transforms.ToTensor()])
        x = transform(x)

        if self.phase == 'train':
            sample = dict(
                category=category,
                cate=category_idx[category],
                idx=idx,
                rot_mat=so3_index,
                #rot_mat_all = so3,
                img=x,
            )
        else:
            sample = dict(
                category=category,
                cate=category_idx[category],
                idx=idx,
                rot_mat=so3_index,
                rot_mat_all=so3,
                img=x,
            )
        return sample


def get_dataloader_symsol(phase, config):
    if phase == 'train':
        batch_size = config.batch_size
        shuffle = True

    elif phase == 'test':
        batch_size = config.batch_size // torch.cuda.device_count()
        shuffle = False

    else:
        raise ValueError

    if config.category_num == 1:
        category = config.category
        dset = SymsolDataset(phase, config, category)
        dloader = DataLoader(dset, batch_size=config.batch_size,
                             shuffle=shuffle, num_workers=config.num_workers, pin_memory=True)
        return dloader
    else:
        datasets = []
        for category in symsol1:
            dset = SymsolDataset(phase, config, category)
            datasets.append(dset)
        entire_dataset = MixDataset(datasets)
        entire_dloader = DataLoader(entire_dataset, batch_size=batch_size,
                                    num_workers=config.num_workers, shuffle=shuffle, pin_memory=True)
        return entire_dloader, [DataLoader(cat_dset, batch_size=batch_size, num_workers=config.num_workers, shuffle=shuffle, pin_memory=True) for cat_dset in datasets], symsol1
