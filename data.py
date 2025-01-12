import torch
import numpy as np
from torch.utils.data import Dataset, random_split
import h5py 
import os
import pandas as pd
from PIL import Image


class NYUDepth(Dataset):
    def __init__(self, csv_path, transform_image=None, transform_depth=None, transform_pair=None):

        with open(csv_path, "r") as file:
            self.file_paths = [line.strip() for line in file.readlines()][1:]
        
        self.transform_image = transform_image
        self.transform_depth = transform_depth
        self.transform_pair = transform_pair

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        h5_path = self.file_paths[idx]

        with h5py.File(h5_path, "r") as h5f:
            rgb = np.array(h5f["rgb"]).transpose(1, 2, 0) 
            depth = np.array(h5f["depth"]) 

        rgb = Image.fromarray(rgb)
        depth = Image.fromarray(depth.astype(np.float32))

        if self.transform_pair:
            rgb, depth = self.transform_pair(rgb, depth)
            
        if self.transform_image:
            rgb = self.transform_image(rgb)

        if self.transform_depth:
            depth = self.transform_depth(depth)

        return rgb, depth


class COCO(Dataset):
    def __init__(self, csv_path, base_path='./data/', transform_image=None, transform_depth=None, transform_pair=None):
        self.base_path = base_path
        self.transform_image = transform_image
        self.transform_depth = transform_depth
        self.transform_pair = transform_pair


        self.data_frame = pd.read_csv(csv_path, sep=";", header=None, names=["image_path", "depth_path"])


        self.data_frame["image_path"] = self.data_frame["image_path"].apply(lambda x: os.path.join(base_path, x))
        self.data_frame["depth_path"] = self.data_frame["depth_path"].apply(lambda x: os.path.join(base_path, x))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        img_path = self.data_frame.iloc[idx]["image_path"]
        depth_path = self.data_frame.iloc[idx]["depth_path"]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        image = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)

        if depth.mode == "RGB":
            depth = depth.convert("L")  

        if self.transform_pair:
            image, depth = self.transform_pair(image, depth)

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_depth:
            depth = self.transform_depth(depth)

        return image, depth


def load_nyu_train(transform_image=None, transform_depth=None, transform_pair=None):
    train_data =   NYUDepth(
                    csv_path='./data/nyudepthv2/train_files.csv',
                    transform_image=transform_image,
                    transform_depth=transform_depth,
                    transform_pair=transform_pair
                    )
    train_size = int(0.95 * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, val_subset = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    return train_subset, val_subset

def load_nyu_test(transform_image=None, transform_depth=None, transform_pair=None):
    return  NYUDepth(
            csv_path='./data/nyudepthv2/test_files.csv',
            transform_image=transform_image,
            transform_depth=transform_depth,
            transform_pair=transform_pair
            )
    
def load_coco_train(transform_image=None, transform_depth=None, transform_pair=None):
    train_data =    COCO(
                        csv_path='./data/COCO/COCO.csv',
                        transform_image=transform_image,
                        transform_depth=transform_depth,
                        transform_pair=transform_pair
                    )
    train_size = int(0.95 * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, val_subset = random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    return train_subset, val_subset 