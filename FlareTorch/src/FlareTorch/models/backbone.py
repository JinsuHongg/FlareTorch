import torch
from torch import nn
import torchvision.models as models


class ResNet18Regressor(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(ResNet18Regressor, self).__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(weights=None)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.resnet.conv1 = nn.Conv2d(
            merged_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input: B, C, T, H, W
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through ResNet
        features = self.resnet(x_merged)  # B, 512, H', W'

        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # B, 512

        # Classification
        features = self.dropout(features)
        output = self.regressor(features)

        return output.squeeze(-1)


class ResNet34Regressor(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(ResNet34Regressor, self).__init__()
        # Load pretrained ResNet34
        self.resnet = models.resnet34(weights=None)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.resnet.conv1 = nn.Conv2d(
            merged_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input: B, C, T, H, W
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through ResNet
        features = self.resnet(x_merged)  # B, 512, H', W'

        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # B, 512

        # Classification
        features = self.dropout(features)
        output = self.regressor(features)

        return output.squeeze(-1)


class ResNet50Regressor(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(ResNet50Regressor, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=None)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.resnet.conv1 = nn.Conv2d(
            merged_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Input: B, C, T, H, W
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through ResNet
        features = self.resnet(x_merged)  # B, 2048, H', W'

        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # B, 2048

        # Classification
        features = self.dropout(features)
        output = self.regressor(features)

        return output.squeeze(-1)


class AlexNetRegressor(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(AlexNetRegressor, self).__init__()
        # Load pretrained AlexNet
        self.alexnet = models.alexnet(weights=None)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.alexnet.features[0] = nn.Conv2d(
            merged_channels, 64, kernel_size=11, stride=4, padding=2
        )

        # Remove the final classification layer
        self.alexnet = nn.Sequential(*list(self.alexnet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        # Use adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.regressor = nn.Linear(9216, num_classes)  # 6*6*256 = 9216

    def forward(self, x):
        # Input: B, C, T, H, W
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through AlexNet features
        features = self.alexnet(x_merged)  # B, 256, H', W'

        # Adaptive pooling to ensure consistent size
        features = self.adaptive_pool(features)  # B, 256, 6, 6

        # Flatten
        features = features.view(B, -1)  # B, 9216

        # Classification
        features = self.dropout(features)
        output = self.regressor(features)

        return output.squeeze(-1)


class MobileNetRegressor(nn.Module):
    def __init__(self, in_channels=3, time_steps=1, num_classes=1, dropout=0.1):
        super(MobileNetRegressor, self).__init__()
        # Load pretrained MobileNet
        self.mobilenet = models.mobilenet_v2(weights=None)

        merged_channels = in_channels * time_steps
        # Modify first conv layer to handle merged channels
        self.mobilenet.features[0][0] = nn.Conv2d(
            merged_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Remove the final classification layer
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])

        # Add classification layers
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(1280, num_classes)

    def forward(self, x):
        # Input: B, C, T, H, W
        B, C, T, H, W = x.shape

        # Merge T and C channels: B, C*T, H, W
        x_merged = x.permute(0, 2, 1, 3, 4).contiguous()  # B, T, C, H, W
        x_merged = x_merged.view(B, C * T, H, W)  # B, C*T, H, W

        # Pass through MobileNet
        features = self.mobilenet(x_merged)  # B, 1280, H', W'

        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # B, 1280

        # Classification
        features = self.dropout(features)
        output = self.regressor(features)

        return output.squeeze(-1)
