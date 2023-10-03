'''Module for processing dMRI data'''

import math
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import einops as ein

from skimage.util import view_as_windows
from npy_patcher import PatcherFloat  # pylint: disable=no-name-in-module


def get_padding(
    orig_shape: Tuple[int, ...],
    patch_shape: Tuple[int, ...],
    strides: Optional[Tuple[int, ...]] = None,
) -> Tuple[Tuple[int, int], ...]:
    '''Calculates padding needed to ensure whole 3D volume
        can be sliced into 3D patches of shape `patch_shape`.

    Args:
        orig_shape: e.g in 3D: (i, j, k)
        patch_shape: e.g. in 3D (m, n, o)
        strides: e.g. in 3D: (s1, s2, s3)

    Returns:
        padding: Nested tuple of padding
            length (pad1, pad2) for each spatial dimension
    '''
    if strides is None:
        strides = patch_shape
    else:
        warnings.warn(
            'get_padding will calculate the padding needed such that all '
            'input is seen when slicing into patches. '
            '\nNOTE: This is different from "valid" and "same" padding.'
        )

    padding = ()
    for idx, size in enumerate(orig_shape):
        patch_size = patch_shape[idx]
        stride = strides[idx]

        # If length of dimension is less than or equal to the patch size then the padding
        # required will be the amount needed to bring the length of the data to the
        # patch size.
        if size <= patch_size:
            pad = patch_size - size
        else:
            num = math.ceil((size - patch_size) / stride)
            pad = ((num * stride) + patch_size) - size

        if pad == 0:
            padding += ((0, 0),)
        else:
            total_pad = pad
            if total_pad % 2 == 0:
                padding += ((total_pad // 2, total_pad // 2),)
            else:
                padding += (((total_pad // 2) + 1, total_pad // 2),)

    return padding


def get_same_padding(patch_shape: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    '''Same padding, only implemeneted for stride of 1'''
    padding = []
    for size in patch_shape:
        total = size - 1
        left = total // 2
        right = total - left
        padding.append((left, right))
    return tuple(padding)


def apply_padding(data_array: np.ndarray, padding: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    '''Applies padding to data array.

    Args:
        data_array: array containing data
            with data_array.ndim >= 3 and first three
            dimensions correspond to (i, j, k)
        padding: Nested tuple of padding
            length (inner, outer) for each spatial dimension

    Returns:
        data_array: modified array
            with dimensions of (i+di, j+dj, k+dk, ...)
    '''
    # Get extra dims padding
    extra_dim_pads = tuple(((0, 0) for _ in data_array.shape[3:]))

    # Apply padding
    data_array = np.pad(data_array, padding + extra_dim_pads)

    return data_array


class TrainingPatcher:
    '''Wrapper class for C++ patcher object'''

    def __init__(self, patch_shape: Tuple[int, ...]) -> None:
        self._patcher = PatcherFloat()
        self.patch_shape = patch_shape

    def get_patch(self, fod_fpath: str, pnum: int) -> np.ndarray:
        '''Loads patch from disk'''
        # pylint: disable=arguments-differ
        patch = self._patcher.get_patch(
            fod_fpath,
            tuple(range(45)),
            self.patch_shape,
            self.patch_shape,
            pnum,
        )
        full_patch_shape = (45,) + self.patch_shape
        patch = np.array(patch, dtype=np.float32).reshape(full_patch_shape)
        return patch

    def get_patch_index(self, mask: np.ndarray) -> np.ndarray:
        '''Calculates patch index for entire subject'''
        i, j, k = self.patch_shape
        padding = get_padding(mask.shape, self.patch_shape)
        mask = apply_padding(mask, padding)
        mask = ein.rearrange(mask, '(ix i) (jx j) (kx k) -> (ix jx kx) i j k', i=i, j=j, k=k)
        mask_filter = np.sum(mask, (1, 2, 3), dtype=bool)
        patch_index = np.arange(len(mask), dtype=np.int32)[mask_filter]

        return patch_index


class PredictionPatcher:
    '''Patcher for FODNet Prediction processing'''

    @staticmethod
    def _extract_unused_voxels(fod_lr: Path, mask_filter: np.ndarray, orig_shape: Tuple[int, ...]):
        '''Extracts unused voxels from npy file'''
        fod: np.ndarray = np.load(fod_lr, allow_pickle=False)
        fod = fod.transpose(1, 2, 3, 0)
        fod = fod.reshape(math.prod(orig_shape), 45)
        fod = fod[~mask_filter, :]
        return fod

    @staticmethod
    def _get_mask_filter(mask: np.ndarray, patch_shape: Tuple[int, ...]):
        # pylint: disable=arguments-differ

        # Get padding
        padding = get_same_padding(patch_shape)

        # Pad mask
        mask = apply_padding(mask, padding)

        # Rearrange mask
        mask = view_as_windows(mask, patch_shape, 1)
        mask = ein.rearrange(mask, 'mx nx ox m n o -> (mx nx ox) m n o')

        # Filter out patches that are not contained within brain mask
        mask_filter = np.sum(mask, (1, 2, 3), dtype=bool)

        # Apply filer to mask
        mask = mask[mask_filter, ...]

        return mask, mask_filter, padding

    @staticmethod
    def _get_filter_order(mask_filter: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx_pos = np.arange(len(mask_filter))[mask_filter]
        idx_neg = np.arange(len(mask_filter))[~mask_filter]
        order = np.argsort(np.concatenate([idx_neg, idx_pos]))

        return order, idx_neg

    @classmethod
    def forward(
        cls,
        dataset: Dict[str, Any],
        context: Dict[str, Any],
        patch_shape: Tuple[int, int, int] = (9, 9, 9),
    ):
        '''Slices data into patches with a stride of 1 in each dim.

        Args:
            datasets (Dict[str,Any]):
                'mask': (np.ndarray) -> shape (i, j, k)
                ...
            context (Dict[str,Any]):
                ...
            patch_shape: Patch shape

        Modifies:
            datasets (Dict[str,Any]):
                + 'mask_filter': (np.ndarray) -> shape (N,)
                + 'padding': (Tuple[Tuple[int,int], ...])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)
                ...

            context (Dict[str,Any]):
                + 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                ...
        '''
        # pylint: disable=invalid-name
        print('Slicing data into 3D patches...')
        context['orig_shape'] = dataset['mask'].shape

        # Get mask filter & slice
        _, dataset['mask_filter'], dataset['padding'] = cls._get_mask_filter(
            dataset['mask'], patch_shape
        )

    @classmethod
    def backward(cls, dataset: Dict[str, Any], context: Dict[str, Any]):
        '''Combines 3D patches into 3D whole volumes

        Args:
            datasets (Dict[str,Any]):
                'fod_lr': (Path) -> filepath to fod lowres
                'fod_hr': (np.ndarray) -> shape (X, 45)
                'mask_filter': (np.ndarray) -> shape (N,)
                'padding': (Tuple[Tuple[int,int,int]])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)

            context (Dict[str,Any]):
                'orig_shape': (Tuple[int,int,int]) -> i, j, k
                ...

        Modifies:
            datasets (Dict[str,Any]):
                ~ 'fod_hr': (np.ndarray) -> shape (i, j, k, 45)
                - 'mask_filter': (np.ndarray) -> shape (N,)
                - 'padding': (Tuple[Tuple[int,int,int]])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)

            context (Dict[str,Any]):
                - 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                ...
        '''
        print('Combining 3D patches into contiguous volumes...')
        orig_shape, mask_filter = context.pop('orig_shape'), dataset.pop('mask_filter')
        del dataset['padding']

        # extract original voxels
        unused_voxels = cls._extract_unused_voxels(dataset['fod_lr'], mask_filter, orig_shape)

        order, _ = cls._get_filter_order(mask_filter)

        # Append real data to original background
        dataset['fod_hr'] = np.concatenate([unused_voxels, dataset['fod_hr']], axis=0)

        # Re-order patches
        dataset['fod_hr'] = dataset['fod_hr'][order, ...]

        # Recombine
        dataset['fod_hr'] = dataset['fod_hr'].reshape(*orig_shape, 45)
