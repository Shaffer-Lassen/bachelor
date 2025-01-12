import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

#Densedepth model from: https://github.com/ialhashim/DenseDepth/blob/master/PyTorch/model.py

class ConvReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, negative_slope=0.1):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x
    
class UpSampleInterpolate(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleInterpolate, self).__init__()
        self.conv_block_a = ConvReLUBlock(skip_input, output_features)
        self.conv_block_b = ConvReLUBlock(output_features, output_features)

    def forward(self, x, skip):
        up_x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)

        concat = torch.cat([up_x, skip], dim=1)

        out = self.conv_block_a(concat)
        out = self.conv_block_b(out)

        return out

class DenseEncoder(nn.Module):
    def __init__(self, weights=None):
        super(DenseEncoder, self).__init__()       
        self.original_model = models.densenet169(weights=weights)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class DenseDecoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width = 1.0):
        super(DenseDecoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSampleInterpolate(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSampleInterpolate(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSampleInterpolate(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSampleInterpolate(skip_input=features//8 + 64,  output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        
        x_d0 = self.conv2(F.relu(x_block4))
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        
        return F.relu(self.conv3(x_d4))

class DenseModel(nn.Module):
    def __init__(self, weights=None):
        super(DenseModel, self).__init__()
        self.encoder = DenseEncoder(weights=weights)
        self.decoder = DenseDecoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )
    
dummy_input = torch.randn(1, 3, 224, 224)
model = DenseModel()
output = model(dummy_input)