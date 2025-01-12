import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_b


class SwinEncoder(nn.Module):
    def __init__(self, weights='DEFAULT'):
        super().__init__()
        swin = swin_b(weights=weights)
        self.patch_embed = swin.features[0]
        self.stage1 = swin.features[1:3]
        self.stage2 = swin.features[3:5]
        self.stage3 = swin.features[5:7]
        self.stage4 = swin.features[7:]
        
    def forward(self, x):
        features = []
        x = self.patch_embed(x)                 
        features.append(x)
        for layer in self.stage1:
            x = layer(x)
        features.append(x)
        for layer in self.stage2:
            x = layer(x)
        features.append(x)
        for layer in self.stage3:
            x = layer(x)
        features.append(x)
        for layer in self.stage4:
            x = layer(x)
        features.append(x)
        return features

class SwinDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip_conv1 = nn.Conv2d(1024, 512, 1)
        self.skip_conv2 = nn.Conv2d(512, 256, 1)
        self.skip_conv3 = nn.Conv2d(256, 128, 1)
        self.skip_conv4 = nn.Conv2d(128, 64, 1)
        self.conv1 = nn.Conv2d(1536, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(768, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(384, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(192, 64, 3, padding=1)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, features):
        features = [x.permute(0, 3, 1, 2) for x in features]
        x = features[-1]
        x = F.interpolate(x, size=features[-2].shape[2:], mode='bilinear', align_corners=False)
        skip = self.skip_conv1(features[-2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = F.interpolate(x, size=features[-3].shape[2:], mode='bilinear', align_corners=False)
        skip = self.skip_conv2(features[-3])
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        x = F.interpolate(x, size=features[-4].shape[2:], mode='bilinear', align_corners=False)
        skip = self.skip_conv3(features[-4])
        x = torch.cat([x, skip], dim=1)
        x = self.conv3(x)
        x = F.interpolate(x, size=features[-5].shape[2:], mode='bilinear', align_corners=False)
        skip = self.skip_conv4(features[-5])
        x = torch.cat([x, skip], dim=1)
        x = self.conv4(x)
        x = self.final_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return F.relu(x)

class SwinModel(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.encoder = SwinEncoder(weights=weights)
        self.decoder = SwinDecoder()
        
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output