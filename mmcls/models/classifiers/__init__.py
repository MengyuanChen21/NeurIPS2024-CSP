# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .vit_lt import VitClassifier
from .multi_modal import CLIPScalableClassifier
from .align import AlignClassifier
from .altclip import AltCLIPClassifier
from .groupvit import GroupViTClassifier
from .openclip import OpenClipClassifier
# HuggingFaceCLIPScalableClassifier, HuggingFaceSimpleScalableClassifier, CLIPZOCClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', 'VitClassifier', 'CLIPScalableClassifier', 'AlignClassifier',
           'AltCLIPClassifier', 'GroupViTClassifier', 'OpenClipClassifier']
