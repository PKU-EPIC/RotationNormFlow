from dataset.dataset_pascal import get_dataloader_pascal3d
from dataset.dataset_symsol import get_dataloader_symsol
from dataset.dataset_raw import get_dataloader_raw
from dataset.dataset_modelnet import get_dataloader_modelnet


def get_dataloader(dataset, phase, config):
    if dataset == "modelnet":
        return get_dataloader_modelnet(phase, config)
    elif dataset == "pascal3d":
        return get_dataloader_pascal3d(phase, config)
    elif dataset == "symsol":
        return get_dataloader_symsol(phase, config)
    elif dataset == "raw":
        return get_dataloader_raw(phase, config)
