import os
from os.path import join, basename, dirname, abspath
import json
import logging
import shutil
import csv
import torch
import numpy as np


class TrainClock(object):
    """ Clock object to track epoch and iteration during training
    """

    def __init__(self, schedulers=None):
        self.epoch = 0
        self.minibatch = 0
        self.iteration = 0
        # used for ema
        self.scratch_iter = 0
        self.schedulers = schedulers

    def tick(self):
        self.minibatch += 1
        self.iteration += 1
        # used for ema
        self.scratch_iter += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0
        if self.schedulers != None:
            for scheduler in self.schedulers:
                scheduler.step()

    def make_checkpoint(self):
        clock_dict = {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'iteration': self.iteration
        }
        if self.schedulers != None:
            for i in range(len(self.schedulers)):
                clock_dict[f'scheduler_{i}'] = self.schedulers[i].state_dict()
        return clock_dict

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.iteration = clock_dict['iteration']
        # if self.schedulers != None:
        #     for i in range(len(self.schedulers)):
        #         self.schedulers[i].load_state_dict(clock_dict[f'scheduler_{i}'])


class KSchedule(object):
    """ linear interpolation of k
    """

    def __init__(self, k_init, k_safe, max_iters):
        self.k_init = k_init
        self.k_safe = k_safe
        self.max_iters = max_iters

    def get_k(self, cur_iter):
        ratio = min(cur_iter // (self.max_iters // 10), 9) / 9
        k = self.k_init + ratio * (self.k_safe - self.k_init)
        return k


class Table(object):
    def __init__(self, filename):
        '''
        create a table to record experiment results that can be opened by excel
        :param filename: using '.csv' as postfix
        '''
        assert '.csv' in filename
        self.filename = filename

    @staticmethod
    def merge_headers(header1, header2):
        # return list(set(header1 + header2))
        if len(header1) > len(header2):
            return header1
        else:
            return header2

    def write(self, ordered_dict):
        '''
        write an entry
        :param ordered_dict: something like {'name':'exp1', 'acc':90.5, 'epoch':50}
        :return:
        '''
        if os.path.exists(self.filename) == False:
            headers = list(ordered_dict.keys())
            prev_rec = None
        else:
            with open(self.filename) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                prev_rec = [row for row in reader]
            headers = self.merge_headers(headers, list(ordered_dict.keys()))

        with open(self.filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, headers)
            writer.writeheader()
            if not prev_rec == None:
                writer.writerows(prev_rec)
            writer.writerow(ordered_dict)


class WorklogLogger:
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file,
                            level=logging.DEBUG,
                            format='%(asctime)s - %(threadName)s -  %(levelname)s - %(message)s')

        self.logger = logging.getLogger()

    def put_line(self, line):
        self.logger.info(line)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_args(args, save_dir):
    param_path = os.path.join(save_dir, 'params.json')

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'\nMaking directory: {path}...\n')


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def remkdir(path):
    """
    if dir exists, remove it and create a new one
    :param path:
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def requires_grad(xs, req=False):
    if not (isinstance(xs, tuple) or isinstance(xs, list)):
        xs = tuple(xs)
    for x in xs:
        x.requires_grad_(req)


def dict_get(dict, key, default, default_device='cuda'):
    v = dict.get(key)
    default_tensor = torch.tensor([default]).float().to(default_device)
    if v is None or v.nelement() == 0:
        return default_tensor
    else:
        return v


def acc(x, thres):
    return (x <= thres).sum() / len(x)


def svd(x):
    try:
        return torch.linalg.svd(x)
    except Exception as e:
        print('tensor', x)
        raise e


def assertion(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def geodesic_distance_rotmats(r1s, r2s):
    prod = torch.einsum('nij,nij->n', r1s, r2s)
    angs = torch.acos(torch.clip((prod - 1.)/2., -1., 1.))
    return angs


def min_geodesic_distance_rotmats(r1s, r2s):
    prod = torch.einsum('nij,nkij->nk', r1s, r2s)
    max_prod = torch.max(prod, axis=-1).values
    angs = torch.acos(torch.clip((max_prod - 1.)/2., -1., 1.))
    return angs


def min_geodesic_distance_rotmat(r1s, r2s):
    prod = torch.einsum('nij,kij->nk', r1s, r2s)
    max_prod = torch.max(prod, axis=-1).values
    angs = torch.acos(torch.clip((max_prod - 1.)/2., -1., 1.))
    return angs


def geodesic_distance_rotmats_pairwise(r1s, r2s):
    prod = torch.einsum('nij,mij->nm', r1s, r2s)
    angs = torch.acos(torch.clip((prod - 1.)/2., -1., 1.))
    return angs


def min_geodesic_distance_rotmats_pairwise(r1s, r2s):
    prod = torch.einsum('...nij,...mij->...nm', r1s, r2s)
    max_prod = torch.max(prod, axis=-1).values
    angs = torch.acos(torch.clip((max_prod - 1.)/2., -1., 1.))
    return angs


def find_closest_rot_inds_rotmat(grid_rot, gt_rot):
    if gt_rot.ndim == 2:
        gt_rot = gt_rot[None]
    traces = torch.einsum('gij,lkij->glk', grid_rot, gt_rot)
    max_inds = torch.argmax(traces, axis=0)
    return max_inds


def save_loss_and_kl(loss, kl, config):
    save_loss_path = os.path.join(
        config.model_dir, "{}_loss.npy".format(config.target_fn))
    save_kl_path = os.path.join(
        config.model_dir, "{}_kl.npy".format(config.target_fn))
    np.save(save_loss_path, loss)
    np.save(save_kl_path, kl)


def softplus(x):
    return torch.log(1. + torch.exp(x))
def softplus_inv(x):
    return torch.log(-1. + torch.exp(x))
def softmax(x):
    ex = torch.exp(x - torch.max(x))
    return ex / torch.sum(ex)

