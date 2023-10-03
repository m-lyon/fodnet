'''FODNet Prediction Processor'''

import shutil
import tempfile

from pathlib import Path
from typing import Tuple, Union, Any, Dict, Optional

import numpy as np
import lightning.pytorch as pl

import torch
from torch.utils.data import DataLoader

from fodnet.core.io import load_nifti, save_nifti
from fodnet.core.processing import PredictionPatcher
from fodnet.core.dataset import FODNetPredictDataset


class FODNetPredictionProcessor:
    '''FODNet Test Set Processor'''

    def __init__(self, batch_size=4, num_workers=8, accelerator='gpu'):
        '''Initializes processor object

        Args:
            batch_size (int): Batch size for prediction
            num_workers (int): Number of CPU workers for dataloader
            accelerator (str): Accelerator to use for prediction, either 'gpu' or 'cpu'.
        '''
        self.bsize = batch_size
        self.patch_shape = (9, 9, 9)
        self.workers = num_workers
        self.accelerator = accelerator
        self._tmpdir = None

    def load_dataset(
        self,
        mask: Union[str, Path],
        fod_lr: Union[str, Path],
        tmp_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        '''Loads dataset into data dict

        Args:
            mask (Union[str,Path]): Path to NIfTI brain mask file
            fod_lr (Union[str,Path]): Path to NIfTI Low-res FOD file
            tmp_dir (Union[str,Path]): Path to temporary directory to save FOD file during
                processing, this should be using an SSD if possible. Defaults to using `tempfile`
                module to create a temporary directory.

        Returns:
            dataset:
                'fod_lr': (Path) -> Path to NumPy FOD file
                'mask': (np.ndarray) -> shape (i, j, k)
            context:
                'affine': (np.ndarray) -> shape (4, 4)
        '''
        # First we resave the NIfTI LR FOD to .npy for use with npy-patcher
        fod, affine = load_nifti(fod_lr)
        if tmp_dir is None:
            self._tmpdir = Path(tempfile.mkdtemp())
            fod_fpath = self._tmpdir.joinpath('fod.npy')
        else:
            fod_fpath = Path(tmp_dir).joinpath('fod.npy')
        np.save(fod_fpath, fod.transpose(3, 0, 1, 2), allow_pickle=False)

        # Load mask into memory to use for patch calculation
        mask_data, _ = load_nifti(mask)

        return {'fod_lr': fod_fpath, 'mask': mask_data}, {'affine': affine}

    def save_dataset(self, datasets, context, out_fpath: Union[str, Path]):
        '''Saves dataset to disk'''
        if self._tmpdir is not None and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir)
        elif datasets['fod_lr'].exists():
            datasets['fod_lr'].unlink()
        save_nifti(datasets['fod_hr'], context['affine'], out_fpath)

    def run_model(self, dataset: Dict[str, Any], model: pl.LightningModule):
        '''Runs model through inference to produce dMRI outputs

        Args:
            datasets:
                'fod_lr': (Path) -> Path to NumPy FOD file
                'mask_filter': (np.ndarray) -> shape (N,)
                'padding': (Tuple[Tuple[int,int], ...])
                    shape -> (padi1, padi2), (padj1, padj), (padk1, padk2)
                ...
            model: Pretrained FODNet model

        Modifies:
            datasets:
                + 'fod_hr': (np.ndarray) -> shape (X, 45)
                ...
        '''
        print('Running prediction on data...')
        pred_dataset = FODNetPredictDataset(
            dataset['fod_lr'], dataset['mask_filter'], dataset['padding']
        )
        dataloader = DataLoader(pred_dataset, self.bsize, pin_memory=True, num_workers=self.workers)
        trainer = pl.Trainer(accelerator=self.accelerator, devices=1, logger=False)
        fod_hr = trainer.predict(model, dataloaders=dataloader)
        dataset['fod_hr'] = torch.cat(fod_hr, dim=0)  # type: ignore

    def run_subject(
        self,
        model: pl.LightningModule,
        mask: Union[str, Path],
        fod_lr: Union[str, Path],
        out_fpath: Union[str, Path],
        tmp_dir: Optional[Union[str, Path]] = None,
    ):
        '''Runs subject through preprocessing, model inference, and postprocessing.
        Optionally save data to disk.

        Args:
            model (torch.nn.Module): Initialised PyTorch model
            fpaths (Tuple[str,str,str,str]): Tuple of dMRI fpath data.
                (dmri_fpath, bvec_fpath, bval_fpath, mask_fpath)
            save_str (str): Path to save combined data, should be a directory path
                and a file prefix.
                Default: Do not save (None).

        Returns:
            datasets (Dict[str,Any]):
                'dmri': (np.ndarray) -> shape (i, j, k, b)
                'bvec': (np.ndarray) -> shape (3, b)
                'bval': (np.ndarray) -> shape (b,)
                'mask': (np.ndarray) -> shape (i, j, k)
            context (Dict[str,Any]):
                'affine': (np.ndarray) -> shape (4, 4)
                'shell_keys': (np.ndarray) -> shape (b,)
        '''
        # Load data
        dataset, context = self.load_dataset(mask, fod_lr, tmp_dir)
        # Preprocess
        PredictionPatcher.forward(dataset, context, patch_shape=self.patch_shape)
        # Run the model
        self.run_model(dataset, model)
        # Postprocessing
        PredictionPatcher.backward(dataset, context)
        # Save to disk
        self.save_dataset(dataset, context, out_fpath)
