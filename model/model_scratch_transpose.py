import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MultiHeadSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        head_dim = in_channels // num_heads

        self.qkv = nn.Conv2d(in_channels, in_channels*3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.scale = head_dim ** -0.5

    def forward(self, x):
        N, C, H, W = x.shape

        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, C, dim=1)

        head_dim = C // self.num_heads
        q = q.view(N, self.num_heads, head_dim, H*W)
        k = k.view(N, self.num_heads, head_dim, H*W)
        v = v.view(N, self.num_heads, head_dim, H*W)

        attn_scores = torch.matmul(
            q.transpose(-2, -1),
            k
        ) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_probs, v.transpose(-2, -1))
        out = out.transpose(-2, -1).contiguous()
        out = out.view(N, C, H, W)

        out = self.proj(out)
        return out


class ConvAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_heads=4):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, stride=stride)
        self.attn = MultiHeadSpatialAttention(out_channels, num_heads=num_heads)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = self.conv_block(x)
        out_attn = self.attn(out)
        out = out + out_attn
        out = out + identity
        return out


class ScratchEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_heads=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.stage1 = ConvAttnBlock(base_channels, base_channels*2, stride=2, num_heads=num_heads)
        self.stage2 = ConvAttnBlock(base_channels*2, base_channels*4, stride=2, num_heads=num_heads)
        self.stage3 = ConvAttnBlock(base_channels*4, base_channels*8, stride=2, num_heads=num_heads)

    def forward(self, x):
        x0 = x
        x1 = self.stem(x0)
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        return [x0, x1, x2, x3, x4]


class ScratchTransposeDecoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.up_trans3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.up_trans2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.up_trans1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, feats):
        x0, x1, x2, x3, x4 = feats
        y3 = self.up_trans3(x4)
        y3 = torch.cat([y3, x3], dim=1)
        y3 = self.up_conv3(y3)
        y2 = self.up_trans2(y3)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.up_conv2(y2)
        y1 = self.up_trans1(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.up_conv1(y1)
        depth = F.relu(self.out_conv(y1))
        return depth


class ScratchTransposeModel(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_heads=4):
        super().__init__()
        self.encoder = ScratchEncoder(in_channels=in_channels, base_channels=base_channels, num_heads=num_heads)
        self.decoder = ScratchTransposeDecoder(base_channels=base_channels)

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth


if __name__ == "__main__":
    model = ScratchTransposeModel(in_channels=3, base_channels=64, num_heads=4)
    x = torch.randn(1, 3, 224, 224)
    depth_pred = model(x)
    print("Input:", x.shape)
    print("Output depth:", depth_pred.shape)
