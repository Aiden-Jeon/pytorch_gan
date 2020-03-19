import torch

def bce_loss(score, target):
    return torch.nn.BCELoss()(score, target)

def mse_loss(score, target):
    return torch.nn.MSELoss()(score, target)
