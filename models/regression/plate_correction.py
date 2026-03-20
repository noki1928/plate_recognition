import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PlateCorrectionResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        if pretrained:
            for param in list(self.backbone.parameters())[:-20]:
                param.requires_grad = False


        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 8)
        )


    def forward(self, x):
        features = self.backbone(x)
        coords = self.regressor(features)
        return coords.view(-1, 4, 2)
