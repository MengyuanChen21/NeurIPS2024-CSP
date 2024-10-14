import torch
import numpy as np
import os
import torch.nn.functional as F


def print_category(outputs, softmax=False):
    pred = outputs.detach().clone()
    if softmax:
        pred = F.softmax(pred, dim=1)
    pred_cat = torch.argmax(pred, dim=1).cpu().tolist()
    if os.environ['LOCAL_RANK'] == '0':
        print(pred_cat)

def print_topk(outputs, softmax=False, k=3):
    pred = outputs.detach().clone()
    if softmax:
        pred = F.softmax(pred, dim=1)
    _, pred_cat = torch.topk(pred, k, dim=1)
    pred_cat = pred_cat.cpu().tolist()
    if os.environ['LOCAL_RANK'] == '0':
        print(pred_cat)

def add_noise(target, std_ratio):
    std_target = target.std()
    std_noise = std_ratio * std_target
    noise = torch.randn_like(target) * std_noise
    target += noise
    target[target < 0] = 0
    target = target / target.sum()
    return target
