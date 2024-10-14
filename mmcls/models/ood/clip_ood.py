from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter
import json
import random

from ..builder import OOD
from mmcls.models import build_classifier
from .utils import add_noise

@OOD.register_module()
class ScalableClassifier(BaseModule):
    def __init__(self, classifier, t=1, ngroup=20, group_fuse_num=None, dump_score=None, **kwargs):
        super(ScalableClassifier, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.dump_score = dump_score
        self.t = t
        self.ngroup = ngroup
        self.group_fuse_num = group_fuse_num

    def grouping(self, pos, neg, num, ngroup=10, random_permute=False):
        group = ngroup
        drop = neg.shape[1] % ngroup
        if drop > 0:
            neg = neg[:, :-drop]
        if random_permute:
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg.shape[1], device="cuda:{}".format(self.local_rank))
            neg = neg.T
            negs = neg[idx].T.reshape(pos.shape[0], group, -1).contiguous()
        else:
            negs = neg.reshape(pos.shape[0], group, -1).contiguous()
        scores = []
        for i in range(group):
            full_sim = torch.cat([pos, negs[:, i, :]], dim=-1) / self.t
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        if num is not None:
            scores = scores[:,0:num-1]
        score = scores.mean(dim=-1)
        return score

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        if "dataset_name" in input and self.dump_score:
            dataset_name = input['dataset_name']
            dump_path = os.path.join(self.dump_score, "{}".format(dataset_name))
            os.makedirs(dump_path, exist_ok=True)
            del input['dataset_name']
        with torch.no_grad():
            pos, neg = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            full_sim = torch.cat([pos, neg], dim=-1) / self.t

            if self.ngroup > 1:
                score = self.grouping(pos, neg, num=self.group_fuse_num, ngroup=self.ngroup, random_permute=True)
            else:
                # neg_score = torch.topk(full_sim[:, pos.shape[1]:], k=10000, dim=-1).values
                # score = torch.cat([pos, neg_score], dim=-1)
                # score = score.softmax(dim=-1)[:, :pos.shape[1]].sum(dim=-1)
                full_sim = full_sim.softmax(dim=-1)
                # pos_score = full_sim[:, :pos.shape[1]].max(dim=1)[0]
                pos_score = full_sim[:, :pos.shape[1]].sum(dim=1)
                # neg_score = full_sim[:, pos.shape[1]:].max(dim=1)[0]
                # pos_score = full_sim[:, :pos.shape[1]]
                # neg_score = torch.topk(full_sim[:, pos.shape[1]:], k=pos.shape[1], dim=-1).values
                # score = torch.cat((pos_score, neg_score), dim=1)
                # score = score.softmax(dim=-1)
                # score = score[:, :pos.shape[1]].sum(dim=-1)
                # score = pos_score - neg_score
                score = pos_score
                # score = torch.softmax(full_sim, dim=1)[:, :pos.shape[1]].sum(dim=-1)

        return score, type

