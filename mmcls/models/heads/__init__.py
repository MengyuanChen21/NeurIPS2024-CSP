# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .deit_head import DeiTClsHead
from .linear_head import LinearClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead


__all__ = [
    'ClsHead', 'LinearClsHead', 'MultiLabelClsHead', 'MultiLabelLinearClsHead', 'DeiTClsHead', 'ConformerHead',
]
