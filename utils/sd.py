from tkinter.ttk import Sizegrip
import torch
import numpy as np
import pytorch3d.transforms as pttf
import healpy as hp
from absl import logging
from scipy.spatial.transform import Rotation



def generate_queries(number_queries, mode='random'):
    '''Generate query rotaitons from SO(3)

    Args:
            number_queries: The number of queries.
            mode: 'random' or 'gird'; determines whether to generate rotations from the 
                    uniform distribution over SO(3), or use an equivolumetric grid.

    Returns:
            A tensor of rotation matrices, shape [number_queries, 3, 3]
    '''
    if mode == 'random':
        return pttf.random_rotations(number_queries)
    elif mode == 'grid':
        return get_closest_available_grid(number_queries)


_grids = {}


def get_closest_available_grid(num_queries):
    grid_size = 72 * 8**np.arange(9)
    size = grid_size[np.argmin(
        np.abs(np.log(num_queries) - np.log(grid_size)))]
    if _grids.get(size) is not None:
        return _grids[size]
    else:
        logging.info('Using grid of size %d, Requested was %d',
                     size, num_queries)

        grid_created = False

        if not grid_created:
            _grids[size] = generate_healpix_grid(size=size)
        return _grids[size]


def generate_healpix_grid(recursion_level=None, size=None):
    '''Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).

    Args:
            recursion_level: An integer which determines the level of resolution of the
  grid.  The final number of points will be 72*8**recursion_level.  A
  recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
  for evaluation.
    size: A number of rotations to be included in the grid.  The nearest grid
  size in log space is returned.
    Returns:
    (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
    '''
    # See: # https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L380.
    # See: https://github.com/airalcorn2/pytorch-ipdf/blob/master/evaluate.py
    assert not (recursion_level is None and size is None)
    if size:
        recursion_level = max(
            int(np.round(np.log(size / 72.0) / np.log(8.0))), 0)

    number_per_side = 2**recursion_level
    number_pix = hp.nside2npix(number_per_side)
    s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = np.stack([*s2_points], 1)

    azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
    polars = np.arccos(s2_points[:, 2])
    tilts = np.linspace(0, 2 * np.pi, 6 * 2**recursion_level, endpoint=False)

    R1s = Rotation.from_euler("X", azimuths).as_matrix()
    R2s = Rotation.from_euler("Z", polars).as_matrix()
    R3s = Rotation.from_euler("X", tilts).as_matrix()

    Rs = np.einsum("bij,tjk->tbik", R1s @ R2s, R3s).reshape(-1, 3, 3)
    return torch.Tensor(Rs)
