import torch
from PIL import Image
from torchvision import transforms
from utils import normalize_depth
import random
from torchvision.transforms.functional import hflip
from torchvision.transforms import functional as F


class PairedTransform:
    def __init__(self, flip_prob=0.5, resize_size=256, crop_size=224, mode='train'):
        self.flip_prob = flip_prob
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.mode = mode

    def __call__(self, image: Image.Image, depth: Image.Image):
        image = F.resize(image, self.resize_size, interpolation=transforms.InterpolationMode.BILINEAR)
        depth = F.resize(depth, self.resize_size, interpolation=transforms.InterpolationMode.NEAREST)

        if self.mode == 'train':
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.crop_size, self.crop_size)
            )
            image = F.crop(image, i, j, h, w)
            depth = F.crop(depth, i, j, h, w)
        else:
            image = F.center_crop(image, self.crop_size)
            depth = F.center_crop(depth, self.crop_size)

        if self.mode == 'train' and random.random() < self.flip_prob:
            image = F.hflip(image)
            depth = F.hflip(depth)

        return image, depth


class PrepareImageTransform:
    def __init__(self, image_size=224, mean=None, std=None, mode='train'):
        self.mode = mode
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
        
        self.transform = transforms.Compose(transforms_list)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class PrepareDepthTransform:
    def __init__(self, image_size=224, mode='train', dataset='NYUv2', normalize_zero_one=False):
        self.image_size = image_size
        self.mode = mode
        self.dataset = dataset
        self.normalize_zero_one = normalize_zero_one

        if mode == 'train':
            target_size = (image_size//2, image_size//2)
            self.transform = transforms.Compose([
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
    def __call__(self, depth_map: Image.Image) -> torch.Tensor:
        depth_tensor = self.transform(depth_map)

        if self.dataset == 'COCO':
            depth_tensor = normalize_depth(depth_tensor)
        
        if self.normalize_zero_one:
            depth_tensor = normalize_depth(depth_tensor)
            depth_tensor = 1.0 - depth_tensor

        return depth_tensor