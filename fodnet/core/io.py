'''NIfTI I/O functions.'''

from pathlib import Path
from typing import Union, Type, Tuple

import numpy as np
import nibabel as nib
from nibabel.loadsave import load, save
from nibabel.funcs import as_closest_canonical
from nibabel.orientations import aff2axcodes


def load_nifti(
    nifti_fpath: Union[str, Path], dtype: Type = np.float32, force_ras=False
) -> Tuple[np.ndarray, np.ndarray]:
    '''Loads NIfTI image into memory

    Args:
        nifti_fpath (str): Filepath to nifti image
        dtype (type): Datatype to load array with.
            Default: `np.float32`
        force_ras (bool): Forces data into RAS data ordering scheme.
            Default: `False`.

    Returns:
        data (np.ndarray): image data
        affine (np.ndarray): affine transformation -> shape (4, 4)
    '''
    img: nib.nifti2.Nifti2Image = load(nifti_fpath)  # type: ignore
    if force_ras:
        if aff2axcodes(img.affine) != ('R', 'A', 'S'):
            print(f'Converting {img.get_filename()} to RAS co-ords')
            img = as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=dtype)

    return data, img.affine


def save_nifti(data, affine, fpath, descrip=None):
    '''Saves NIfTI image to disk.

    Args:
        data (np.ndarray): Data array
        affine (np.ndarray): affine transformation -> shape (4, 4)
        fpath (str): Filepath to save to.
        descrip (str): Additional info to add to header description
            Default: `None`.
    '''
    img = nib.nifti2.Nifti2Image(data, affine)

    if descrip is not None:
        img.header['descrip'] = descrip  # type: ignore

    save(img, fpath)
