import torch.nn as nn


def make_loss(pos_weight):
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
