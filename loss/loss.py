from typing import Callable, Union, Any, Optional
import torch
import torch.nn.functional as F


def bce_loss_with_logit(score, target):
    prob = torch.sigmoid(score)
    return F.binary_cross_entropy(prob, target)

def mse_loss(score, target):
    return F.mse_loss(score, target)

def hinge_loss(score):
    return F.relu(score).mean()
        