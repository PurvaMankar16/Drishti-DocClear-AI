import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = ResNet18_Weights.DEFAULT
        except AttributeError:
            weights = None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
