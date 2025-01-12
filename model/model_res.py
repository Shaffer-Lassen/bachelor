import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, resnet152


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


class ResEncoder(nn.Module):
    def __init__(self, resnet=resnet50, weights=None):
        super().__init__()
        encoder = resnet(weights=weights)
        
        self.layer0 = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool
        )
        self.layer1 = encoder.layer1  
        self.layer2 = encoder.layer2  
        self.layer3 = encoder.layer3  
        self.layer4 = encoder.layer4 

        self.reduce_f4 = ConvBatchnormReLU(2048, 512, kernel_size=1) 
        self.reduce_f3 = ConvBatchnormReLU(1024, 256, kernel_size=1)
        self.reduce_f2 = ConvBatchnormReLU(512, 128, kernel_size=1)  
        self.reduce_f1 = ConvBatchnormReLU(256, 64, kernel_size=1)    

    def forward(self, x):
        f1 = self.layer0(x)
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        
        f5 = self.reduce_f4(f5)
        f4 = self.reduce_f3(f4)
        f3 = self.reduce_f2(f3)
        f2 = self.reduce_f1(f2)

        return f1, f2, f3, f4, f5


class ResDecoder(nn.Module):
    def __init__(self, output_channels=1):
        super().__init__()
        self.upsample1 = nn.Sequential(
            ConvBatchnormReLU(512 + 256, 256, kernel_size=3, padding=1),
            ConvBatchnormReLU(256, 256, kernel_size=3, padding=1)
        )

        self.upsample2 = nn.Sequential(
            ConvBatchnormReLU(256 + 128, 128, kernel_size=3, padding=1),
            ConvBatchnormReLU(128, 128, kernel_size=3, padding=1)
        )

        self.upsample3 = nn.Sequential(
            ConvBatchnormReLU(128 + 64, 64, kernel_size=3, padding=1),
            ConvBatchnormReLU(64, 64, kernel_size=3, padding=1)
        )

        self.final_upsample = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        )

    def forward(self, features):
        f1, f2, f3, f4, f5 = features

        def upsample_and_concat(x, concat_with):
            up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
            concat_x = torch.cat([up_x, concat_with], dim=1)
            return concat_x

        x = self.upsample1(upsample_and_concat(f5, f4))
        x = self.upsample2(upsample_and_concat(x, f3))
        x = self.upsample3(upsample_and_concat(x, f2))
        x = upsample_and_concat(x, f1)

        depth = F.relu(self.final_upsample(x))
        
        return depth



class ResModel(nn.Module):
    def __init__(self, output_channels=1, resnet=resnet50, weights=None):
        super().__init__()
        self.encoder = ResEncoder(resnet=resnet, weights=weights)
        self.decoder = ResDecoder(output_channels=output_channels)

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth
