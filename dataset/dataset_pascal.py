import torch
from torch.utils.data import Dataset, DataLoader
import random
from dataset.Pascal3D import Pascal3D, Pascal3D_render, Pascal3D_all
import dataset.dataloader_utils as dataloader_utils
from dataset.dataloader_utils import MixDataset

all_pascal_classes = [
    "aeroplane",
    "bicycle",
    "boat",
    "bottle",
    "bus",
    "car",
    "chair",
    "diningtable",
    "motorbike",
    "sofa",
    "train",
    "tvmonitor",
]


def get_pascal_datasets(dataset_dir, batch_size, train_all, use_augmentation, voc_train, source, category):
    dataset_real = Pascal3D.Pascal3D(dataset_dir, train_all=train_all,
                                     use_warp=True, voc_train=voc_train, source=source, category=category)
    train_real = dataset_real.get_train(use_augmentation)
    real_sampler = torch.utils.data.sampler.RandomSampler(
        train_real, replacement=False)
    dataset_rendered = Pascal3D_render.Pascal3DRendered(
        dataset_dir, category=category)
    # use 20% of synthetic data for training per epoch
    rendered_size = int(0.2 * len(dataset_rendered))
    rendered_sampler = dataloader_utils.RandomSubsetSampler(
        dataset_rendered, rendered_size)
    dataset_train, sampler_train = dataloader_utils.get_concatenated_dataset(
        [(train_real, real_sampler), (dataset_rendered, rendered_sampler)])
    return dataset_train, dataset_real.get_eval()


def get_dataloader_pascal3d(phase, config):
    if phase == 'train':
        batch_size = config.batch_size
        shuffle = True
    elif phase == 'test':
        batch_size = config.batch_size // torch.cuda.device_count()
        shuffle = False
    else:
        raise ValueError

    datasets = []
    for category in all_pascal_classes:
        train_set, eval_set = get_pascal_datasets(
            config.data_dir, batch_size, True, True, False, "both_pascal", category)
        datasets.append(train_set if phase == 'train' else eval_set)
    entire_dataset = MixDataset(datasets)
    entire_dloader = DataLoader(entire_dataset, batch_size=batch_size,
                                num_workers=config.num_workers, shuffle=shuffle, pin_memory=True)
    return entire_dloader, [DataLoader(cat_dset, batch_size=batch_size, num_workers=config.num_workers, shuffle=shuffle, pin_memory=True) for cat_dset in datasets], all_pascal_classes
