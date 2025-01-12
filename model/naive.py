import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)

class NaiveEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv1 = EncoderBlock(64, 128)
        self.conv2 = EncoderBlock(128, 256)
        self.conv3 = EncoderBlock(256, 512)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return F.relu(x)

class NaiveDecoder(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1):
        super().__init__()
        
        self.decode = nn.Sequential(
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32)
        )
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.decode(x)
        return F.relu(self.final_conv(x))
    
    
class NaiveModel(nn.Module):
    def __init__(self, output_channels=1):
        super().__init__()
        self.encoder = NaiveEncoder()
        self.decoder = NaiveDecoder()
    
    def forward(self, x):
        feature_maps = self.encoder(x)
        depth_map = self.decoder(feature_maps)
        return depth_map