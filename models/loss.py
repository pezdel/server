import torch
import torch.nn.functional as F


def recon_loss(inputs, targets, logvar, mu):
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction='sum')
    return bce_loss.int()
