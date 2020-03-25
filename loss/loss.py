import torch

def bce_loss(score, target):
    return torch.nn.BCELoss()(score, target)

def least_squares_loss(score, target):
    """
    Discriminator: last_act=None
    """
    return torch.nn.MSELoss()(score, target)
