import random
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv import FileClient
from torch.utils.data import Dataset
import torch
import json
from PIL import Image
import torchvision as tv
import os
import copy
from collections import Counter
import imgaug as ia
import imgaug.augmenters as iaa
import pandas as pd
# from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose

class OODBaseDataset(Dataset):
    def __init__(self, name, pipeline, input_size=None, transform='ImageNet', pick_img=None, aug=None, noise_engine=None, len_limit=-1):
        super().__init__()
        self.pipeline = Compose(pipeline)
        self.file_list = []
        self.data_prefix = None
        self.name = name
        # self.resize_size = input_size if input_size is not None else 256
        self.resize_size = input_size if input_size is not None else 224
        self.crop_size = input_size if input_size is not None else 224
        self.pick_img = pick_img

        if transform == 'ImageNet':
            self.transform = tv.transforms.Compose([
                # tv.transforms.Resize(256),
                # tv.transforms.Resize(224),
                tv.transforms.Resize(self.resize_size),
                # tv.transforms.Resize(248, interpolation=tv.transforms.InterpolationMode.BICUBIC),
                # tv.transforms.CenterCrop(224),
                tv.transforms.CenterCrop(self.crop_size),
                # tv.transforms.CenterCrop(self.crop_size),
                # tv.transforms.Resize((480, 480)),
                tv.transforms.ToTensor(),
                # tv.transforms.Normalize([123.675/255, 116.28/255, 103.53/255],
                #                         [58.395/255, 57.12/255, 57.375/255]),
                tv.transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                        [0.26862954, 0.26130258, 0.27577711]),
                # For ALIGN Model
                # tv.transforms.Normalize([0.5, 0.5, 0.5],
                #                         [0.5, 0.5, 0.5]),
            ])
        elif transform == 'Cifar':
            self.transform = tv.transforms.Compose([
                tv.transforms.Resize(32),
                tv.transforms.CenterCrop(32),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2023, 0.1994, 0.2010]),
                # tv.transforms.Normalize([129.304/255, 124.07/255, 112.434/255],
                #                         [68.17/255, 65.392/255, 70.418/255]),
            ])

        self.noise_engine = noise_engine
        self.len_limit = len_limit
        self.data_infos = []
        self.aug = aug

    def parse_datainfo(self):
        random.seed(111)
        random.shuffle(self.file_list)
        if self.pick_img is not None:
            self.file_list = self.pick_img
        for sample in self.file_list:
            info = dict(img_prefix=self.data_prefix)
            sample = os.path.join(self.data_prefix, sample)
            info['img_info'] = {'filename': sample}
            info['filename'] = sample
            info['type'] = 3  # no type
            info['label'] = -1  # no label
            self.data_infos.append(info)

    def __len__(self):
        return self.len_limit if self.len_limit!=-1 else len(self.data_infos)
        # return len(self.file_list)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        # sample = Image.open(os.path.join(results['img_prefix'], results['img_info']['filename']))
        try:
            sample = Image.open(results['img_info']['filename'])
        except:
            print(results)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        if self.aug is not None:
            seq = iaa.Sequential([
                # iaa.imgcorruptlike.Fog(severity=5)
                # iaa.GammaContrast((0.8, 1.2)),
                # iaa.LogContrast(gain=(0.5,0.5)),
                # iaa.MultiplyAndAddToBrightness(mul=(0.5, 0.5), add=(0, 0)),
                # iaa.AddToBrightness((-10, 10))
                iaa.Resize({"height": 64, "width": 64})

            ])
            sample = np.array(sample).astype('uint8')
            sample = seq(images=[sample])[0]
            sample = Image.fromarray(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.noise_engine == "uniform":
            sample = sample + ((torch.rand_like(sample) - 0.5) / 5)
        results['img'] = sample
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_data(idx)


@DATASETS.register_module()
class TxtDataset(OODBaseDataset):
    def __init__(self, name, path, data_ann, pipeline, train_label=None, **kwargs):
        super().__init__(name, pipeline, **kwargs)
        self.data_prefix = path
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_ann = data_ann
        self.train_label = train_label
        with open(self.data_ann) as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        for filename in samples:
            self.file_list.append(filename)
        self.parse_datainfo()


    def parse_datainfo(self):
        random.seed(222)
        random.shuffle(self.file_list)
        if self.pick_img is not None:
            self.file_list = self.pick_img
        if self.train_label is not None:
            train_labels = []
            with open(self.train_label, 'r') as f:
                for line in f.readlines():
                    segs = line.strip().split(' ')
                    train_labels.append(int(segs[-1]))
            train_label_index = Counter(train_labels)

        for sample in self.file_list:
            info = dict(img_prefix=self.data_prefix)
            sample[0] = os.path.join(self.data_prefix, sample[0])
            info['img_info'] = {'filename': sample[0]}
            info['filename'] = sample[0]
            if len(sample) > 1:
                gt_label = int(sample[-1])
            else:
                gt_label = None
            info['label'] = gt_label
            if self.train_label is not None:
                freq = train_label_index[gt_label]
                if freq > 100:
                    info['type'] = 0  # head
                elif freq < 20:
                    info['type'] = 2  # tail
                else:
                    info['type'] = 1  # mid
            else:
                info['type'] = 3  # no type
            self.data_infos.append(info)

@DATASETS.register_module()
class JsonDataset(OODBaseDataset):
    # INaturalist
    def __init__(self, name, path, data_ann, pipeline, **kwargs):
        super().__init__(name, pipeline, **kwargs)
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_prefix = path
        self.data_ann = data_ann
        with open(self.data_ann) as f:
            ann = json.load(f)
        images = ann['images']
        images_dict = dict()
        for item in images:
            images_dict[item['id']] = item['file_name']
        annotations = ann['annotations']
        samples = []
        for item in annotations:
            samples.append(images_dict[item['image_id']])
        for filename in samples:
            self.file_list.append(filename)
        self.parse_datainfo()


@DATASETS.register_module()
class FolderDataset(OODBaseDataset):
    def __init__(self, name, path, pipeline, data_ann=None, **kwargs):
        super().__init__(name, pipeline, **kwargs)
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_prefix = path
        images = os.listdir(path)
        for filename in images:
            self.file_list.append(filename)
        self.parse_datainfo()


@DATASETS.register_module()
class MultiFolderDataset(OODBaseDataset):
    def __init__(self, name, path, pipeline, data_ann=None, **kwargs):
        super().__init__(name, pipeline, **kwargs)
        self.data_prefix = path
        if data_ann is not None:
            with open(data_ann, 'r') as f:
                lines = f.readlines()
                for category_dir in lines:
                    images = os.listdir(os.path.join(path, category_dir.strip()))
                    for filename in images:
                        self.file_list.append(os.path.join(category_dir.strip(), filename))
        else:
            for category_dir in os.listdir(path):
                if category_dir.startswith('.'):
                    continue
                images = os.listdir(os.path.join(path, category_dir.strip()))
                for filename in images:
                    self.file_list.append(os.path.join(category_dir.strip(), filename))
        self.parse_datainfo()


@DATASETS.register_module()
class CsvDataset(OODBaseDataset):
    def __init__(self, name, path, pipeline, data_ann=None, **kwargs):
        super().__init__(name, pipeline, **kwargs)
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_prefix = path
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_prefix, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split'] == 2]

        # self.y_array = self.metadata_df['y'].values
        # self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values

        for filename in self.filename_array:
            self.file_list.append(os.path.join(self.data_prefix,filename))
        self.parse_datainfo()

@DATASETS.register_module()
class ImageNetSuperclass(OODBaseDataset):
    def __init__(self, name, path, pipeline, train_label=None, pick_class=None, mode='exclude',**kwargs):
        super().__init__(name, pipeline, **kwargs)
        self.data_prefix = path
        # self.file_list = glob.glob(os.path.join(path, '*'))\
        super_class_names = ["dogs", "other-mammals", "birds", "reptiles_fish_amphibians", "inverterbrates",
                             # "food_plants_fungi",
                             # "devices", "structures_furnishing", "clothes_covering",
                             # "implements_containers_misc-objects", "vehicles"
                             ]
        if mode == 'exclude':
            self.data_ann = ["/data/csxjiang/meta/superclasses/val_{}.txt".format(x)
                             for x in super_class_names if x != pick_class]
        elif mode == 'include':
            self.data_ann = ["/data/csxjiang/meta/superclasses/val_{}.txt".format(pick_class)]
        self.train_label = train_label

        samples = []
        labels = []
        for ann in self.data_ann:
            with open(ann) as f:
                samples.extend([x.strip().rsplit(' ', 1) for x in f.readlines()])
        self.samples = samples
        for filename, gt_label in self.samples:
            labels.append(gt_label)
        labels_unique = list(set(labels))
        labels_unique.sort()
        label_map = dict()
        for i, idx in enumerate(labels_unique):
            label_map[idx] = i
        self.file_list = [[filename, label_map[gt_label]] for filename, gt_label in self.samples]
        self.parse_datainfo()
    #
    def parse_datainfo(self):
        random.seed(111)
        random.shuffle(self.file_list)
        if self.pick_img is not None:
            self.file_list = self.pick_img
        if self.train_label is not None:
            train_labels = []
            with open(self.train_label, 'r') as f:
                for line in f.readlines():
                    segs = line.strip().split(' ')
                    train_labels.append(int(segs[-1]))
            train_label_index = Counter(train_labels)

        for sample in self.file_list:
            info = dict(img_prefix=self.data_prefix)
            sample[0] = os.path.join(self.data_prefix, sample[0])
            info['img_info'] = {'filename': sample[0]}
            info['filename'] = sample[0]
            gt_label = int(sample[-1])
            info['label'] = gt_label
            if self.train_label is not None:
                freq = train_label_index[gt_label]
                if freq > 100:
                    info['type'] = 0  # head
                elif freq < 20:
                    info['type'] = 2  # tail
                else:
                    info['type'] = 1  # mid
            else:
                info['type'] = 3  # no type
            self.data_infos.append(info)