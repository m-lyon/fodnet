'''Processing dataset classes'''

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Union, List

from pathlib import Path

import numpy as np
import lightning.pytorch as pl

from npy_patcher import PatcherFloat  # pylint: disable=no-name-in-module

from torch.utils.data import Dataset, DataLoader
from fodnet.core.processing import TrainingPatcher


@dataclass
class Subject:
    '''Subject Dataclass'''

    lowres_fod: Union[str, Path]
    highres_fod: Union[str, Path]
    brain_mask: Union[str, Path]

    def check_data(self):
        '''Checks all filepaths exist'''
        for fpath in [self.lowres_fod, self.highres_fod, self.brain_mask]:
            if not Path(fpath).is_file():
                raise OSError(f'{fpath} does not exist.')


class FODNetTrainingDataset(Dataset):
    '''FOD-Net Dataset'''

    def __init__(self, subjects: Union[List[Subject], Tuple[Subject, ...]]):
        '''Initialise Dataset

        Args:
            subjects: Subject dataclasses
        '''
        super().__init__()
        self.patcher = TrainingPatcher((9, 9, 9))
        self.subjects = subjects
        self._total_len = 0
        self._total_index: np.ndarray
        self._init_dataset()

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        local_index = self._total_index[idx]
        subject = self.subject_data[local_index[0]]
        return self._get_patch_data(subject, local_index)

    def _init_dataset(self):
        '''Initialises dataset'''
        self.subject_data = []
        for subject in self.subjects:
            subject.check_data()
        self._load_subject_data()
        self._calculate_patches()
        self._calculate_total_length()
        self._set_total_index()

    def _load_subject_data(self):
        '''Loads bvecs, bvals, & mask data'''
        for subject in self.subjects:
            mask = np.load(subject.brain_mask)
            self.subject_data.append(
                {
                    'lr_fod': str(subject.lowres_fod),
                    'hr_fod': str(subject.highres_fod),
                    'mask': mask,
                }
            )

    def _calculate_patches(self):
        for data in self.subject_data:
            data['patch_index'] = self.patcher.get_patch_index(data.pop('mask'))

    def _calculate_total_length(self):
        for data in self.subject_data:
            self._total_len += len(data['patch_index'])

    def _set_total_index(self):
        total_list = []
        for i, data in enumerate(self.subject_data):
            local_patch_idx = np.arange(len(data['patch_index']), dtype=int)
            subject_idx = np.full(len(local_patch_idx), i, dtype=int)
            total_list.append(np.stack([subject_idx, local_patch_idx], axis=1))
        self._total_index = np.concatenate(total_list, axis=0)

    def _get_patch_data(
        self, subject: Dict[str, Any], local_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Loads patch data into memory and returns data objects'''
        fod_lr_fpath, fod_hr_fpath = subject['lr_fod'], subject['hr_fod']
        pnum = subject['patch_index'][local_index[1]]
        lr_patch = self.patcher.get_patch(fod_lr_fpath, pnum)
        hr_patch = self.patcher.get_patch(fod_hr_fpath, pnum)[:, 4, 4, 4]
        return lr_patch, hr_patch


class FODNetDataModule(pl.LightningDataModule):
    '''FODNet Data Module'''

    def __init__(
        self,
        train_subjects: Union[List[Subject], Tuple[Subject, ...]],
        val_subjects: Union[List[Subject], Tuple[Subject, ...]],
        batch_size: int = 16,
        num_workers: int = 8,
    ):
        super().__init__()
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_dataloader: DataLoader
        self._val_dataloader: DataLoader

    def setup(self, stage=None) -> None:
        '''This is run on each GPU'''
        if stage in (None, 'fit'):
            self._train_dataloader = DataLoader(
                FODNetTrainingDataset(self.train_subjects),
                self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
                drop_last=True,
            )
            self._val_dataloader = DataLoader(
                FODNetTrainingDataset(self.val_subjects),
                self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
                drop_last=True,
            )

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


class FODNetPredictDataset(Dataset):
    '''Dataset for FODNet Prediction'''

    def __init__(
        self,
        fod_lr: Path,
        mask_filter: np.ndarray,
        padding: Tuple[Tuple[int, int], ...],
    ):
        '''Initialises dataloader class

        Args:
            fod_lr: filepath to NumPy FOD
            mask_filter: shape -> (N,)
            strides: strides to extract patches
        '''
        super().__init__()
        self.fod_lr = fod_lr
        self.mask_filter = mask_filter
        self.patch_shape = (9, 9, 9)
        self.padding = tuple(item for tup in padding for item in tup)
        self._patcher = PatcherFloat()
        self._strides = (1,) * len(self.patch_shape)
        self._patch_idx = np.arange(len(self.mask_filter))[mask_filter]
        self._sph_idx = tuple(range(45))
        self._full_patch_shape = (len(self._sph_idx),) + self.patch_shape
        self._len = abs(np.sum(mask_filter.astype(int)))

    def _get_patch(self, idx: int):
        '''Extracts patch from FOD npy file'''
        pnum = self._patch_idx[idx]
        patch = self._patcher.get_patch(
            str(self.fod_lr),
            self._sph_idx,
            self.patch_shape,
            self._strides,
            pnum,
            padding=self.padding,
        )
        patch = np.array(patch, dtype=np.float32).reshape(self._full_patch_shape)
        return patch

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._get_patch(idx)
