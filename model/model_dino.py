import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class DinoEncoder(nn.Module):
    def __init__(self, model_name='facebook/dinov2-base'):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        outputs = self.backbone(x, output_hidden_states=True)
    
        features = []
        hidden_states = [
            outputs.hidden_states[3],
            outputs.hidden_states[6],
            outputs.hidden_states[9],
            outputs.hidden_states[-1]
        ]
        
        for i, hidden_state in enumerate(hidden_states):
            B, N, D = hidden_state.shape
            H = W = int((N-1) ** 0.5)
            x = hidden_state[:, 1:, :].reshape(B, H, W, D)
            x = x.permute(0, 3, 1, 2)
            target_size = 56 // (2**i)
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=True)
            features.append(x)
            
        return features

class DinoDecoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_dim, 512, kernel_size=3, padding=1)
        
        self.up1 = UpSampleBlock(512 + input_dim, 256)
        self.up2 = UpSampleBlock(256 + input_dim, 128)
        self.up3 = UpSampleBlock(128 + input_dim, 64)
        
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(112, 112), mode='bilinear', align_corners=True)
        )
        
        self.conv_final = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        

    def forward(self, features):
        skip3, skip2, skip1, x = features
        
        x = self.conv1(x)

        x = self.up1(x, skip1)
        
        x = self.up2(x, skip2)
        
        x = self.up3(x, skip3)
        
        x = self.up4(x)
        
        depth = self.conv_final(x)
        
        return F.relu(depth)

class DinoModel(nn.Module):
    def __init__(self, model_name='facebook/dinov2-base'):
        super().__init__()
        self.encoder = DinoEncoder(model_name)
        self.decoder = DinoDecoder()

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth

if __name__ == "__main__":
    model = DinoModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)