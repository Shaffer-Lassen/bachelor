import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, resnet152
from model.model_res import ConvBatchnormReLU, ResEncoder 

class ConvBatchnormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ResNoSkipDecoder(nn.Module):
    def __init__(self, output_channels=1):
        super().__init__()
        self.upsample1 = nn.Sequential(
            ConvBatchnormReLU(512, 256, kernel_size=3, padding=1),
            ConvBatchnormReLU(256, 256, kernel_size=3, padding=1)
        )

        self.upsample2 = nn.Sequential(
            ConvBatchnormReLU(256, 128, kernel_size=3, padding=1),
            ConvBatchnormReLU(128, 128, kernel_size=3, padding=1)
        )

        self.upsample3 = nn.Sequential(
            ConvBatchnormReLU(128, 64, kernel_size=3, padding=1),
            ConvBatchnormReLU(64, 64, kernel_size=3, padding=1)
        )

        self.final_upsample = nn.Sequential(
            ConvBatchnormReLU(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, output_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, features):
        _, _, _, _, f5 = features
        
        # Progressive upsampling without skip connections
        x = self.upsample1(f5)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.upsample2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.upsample3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        depth = F.relu(self.final_upsample(x))
        
        return depth


class ResNoSkipModel(nn.Module):
    def __init__(self, output_channels=1, resnet=resnet50, weights=None):
        super().__init__()
        self.encoder = ResEncoder(resnet=resnet, weights=weights)
        self.decoder = ResNoSkipDecoder(output_channels=output_channels)

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth 