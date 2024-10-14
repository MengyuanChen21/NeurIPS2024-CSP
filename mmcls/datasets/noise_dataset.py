import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv import FileClient
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision as tv
import os
import copy
from functools import partial
import random

# from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose


class NoiseDataset(Dataset):
    def __init__(self, name, pipeline, length, img_size=224):
        super().__init__()
        self.pipeline = Compose(pipeline)
        self.data_prefix = None
        self.name = name
        self.length = length
        self.random_engine = None
        self.img_size = img_size
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((img_size, img_size)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([123.675/255, 116.28/255, 103.53/255],
                                    [58.395/255, 57.12/255, 57.375/255]),
        ])

    def parse_datainfo(self):
        self.data_infos = []
        for _ in range(self.length):
            info = {'img_prefix': ""}
            info['img_info'] = {'filename': ""}
            info['type'] = 3 # no type
            self.data_infos.append(info)

    def __len__(self):
        return self.length

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        assert self.random_engine, "Random engine has not been configured!"
        sample = self.random_engine()
        assert type(sample) is np.ndarray, "Expect type(sample) = np.ndarray, but got {}".format(type(sample))
        sample = Image.fromarray(sample)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        results['img'] = sample
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

@DATASETS.register_module()
class NoiseDatasetUniform(NoiseDataset):
    def __init__(self, name, pipeline, length, img_size=480):
        super().__init__(name, pipeline, length, img_size)
        self.random_engine = partial(np.random.randint,
                                     low=0, high=256, size=(img_size, img_size, 3), dtype=np.uint8)
        self.parse_datainfo()

@DATASETS.register_module()
class NoiseDatasetGaussian(NoiseDataset):
    def __init__(self, name, pipeline, length, img_size=480):
        super().__init__(name, pipeline, length, img_size)
        self.parse_datainfo()
        def random_engine():
            r = np.random.normal(123.675, 58.395, size=(self.img_size, self.img_size, 1))
            g = np.random.normal(116.28, 57.12, size=(self.img_size, self.img_size, 1))
            b = np.random.normal(103.53, 57.375, size=(self.img_size, self.img_size, 1))
            sample = np.concatenate([r, g, b], axis=-1)
            assert sample.shape == (self.img_size, self.img_size, 3)
            sample = sample.astype(np.uint8)
            return sample
        self.random_engine = random_engine

@DATASETS.register_module()
class NoiseDatasetColorBand(NoiseDataset):
    def __init__(self, name, pipeline, length, img_size=480, band_length=(10, 100)):
        super().__init__(name, pipeline, length, img_size)
        self.parse_datainfo()
        self.band_length = band_length
        def random_engine():
            r = np.zeros((self.img_size, self.img_size, 1), dtype=np.uint8)
            g = np.zeros((self.img_size, self.img_size, 1), dtype=np.uint8)
            b = np.zeros((self.img_size, self.img_size, 1), dtype=np.uint8)
            ptr = 0
            while ptr < self.img_size:
                step = random.randint(*self.band_length)
                r[ptr: ptr+step, :, :] = random.randint(0, 255)
                g[ptr: ptr+step, :, :] = random.randint(0, 255)
                b[ptr: ptr+step, :, :] = random.randint(0, 255)
                ptr += step
            sample = np.concatenate([r, g, b], axis=-1)
            assert sample.shape == (self.img_size, self.img_size, 3)
            sample = sample.astype(np.uint8)
            return sample
        self.random_engine = random_engine