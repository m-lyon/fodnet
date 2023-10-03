#!/usr/bin/env python3
'''Use this to install module'''
from os import path
from setuptools import setup, find_namespace_packages

version = '1.1.0'
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fodnet',
    version=version,
    description='FOD-Net Reimplementation.',
    author='Matthew Lyon',
    author_email='matthewlyon18@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    license='MIT License',
    install_requires=[
        'torch>=2.0.0',
        'lightning>=2.0.0',
        'numpy',
        'einops',
        'scikit-image',
        'nibabel',
        'npy-patcher',
    ],
    packages=find_namespace_packages(),
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows :: Windows 10',
    ],
)
