import os
import numpy as np
import torch
import pytorch3d.transforms as pttf
from torch.utils.data import Dataset, DataLoader


class RawDataset(Dataset):
    def __init__(self, phase, config):
        self.phase = phase
        self.config = config
        self.root = os.path.join(config.data_dir, 'raw')
        data = np.load(self.root + f'/{config.category}_' + phase + '.npy')

        self.rotation = torch.from_numpy(data)
        self.length = self.rotation.shape[0]

    def __len__(self):
        if self.config.length != 0:
            self.length = self.config.length
        return self.length

    def __getitem__(self, idx):
        rotation = self.rotation[idx]
        sample = dict(
            rot_mat=rotation,
        )
        return sample


def get_dataloader_raw(phase, config):
    if phase == 'train':
        shuffle = True
        batch_size = config.batch_size
    elif phase == 'test':
        shuffle = False
        batch_size = config.batch_size // torch.cuda.device_count()
    else:
        raise ValueError
    dset = RawDataset(phase, config)
    dloader = DataLoader(dset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=config.num_workers, pin_memory=True)
    return dloader
