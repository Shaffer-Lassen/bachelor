import torch
import os
from model.model_dino import DinoModel
from model.model_res import ResModel
from model.model_dense import DenseModel
from model.model_scratch import ScratchModel
from model.model_swin import SwinModel                           
from model.model_res_no_skip import ResNoSkipModel
from model.naive import NaiveModel
from model.model_scratch_transpose import ScratchTransposeModel
import torch.nn as nn
import random
import numpy as np
from torchvision.models import resnet50, resnet101, resnet152


def load_checkpoint(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    return model

def normalize_depth(depth_map):
    min_depth = depth_map.min()
    max_depth = depth_map.max()
    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
    return normalized_depth
    
def initialize_model(model_name, device, pretrained=False):
    print(f"Debug: Initializing model '{model_name}' on device '{device}'")
    if model_name == 'naive':
        model = NaiveModel()
    elif model_name == 'dino':
        model = DinoModel()
    elif model_name == 'scratch':
        model = ScratchModel()
    elif model_name == 'scratch_transpose':
        model = ScratchTransposeModel()
    elif model_name == 'res50':
        if pretrained:
            model = ResModel(resnet=resnet50, weights='DEFAULT')
        else:
            model = ResModel(resnet=resnet50)
    elif model_name == 'res101':
        if pretrained:
            model = ResModel(resnet=resnet101, weights='DEFAULT')
        else:
            model = ResModel(resnet=resnet101)
    elif model_name == 'res152':
        if pretrained:
            model = ResModel(resnet=resnet152, weights='DEFAULT')
        else:
            model = ResModel(resnet=resnet152)
    elif model_name == 'res_no_skip':
        if pretrained:
            model = ResNoSkipModel(weights='DEFAULT')
        else:
            model = ResNoSkipModel()
    elif model_name == 'dense':
        if pretrained:
            model = DenseModel(weights='DEFAULT')
        else:
            model = DenseModel()
    elif model_name == 'swin':
        if pretrained:
            model = SwinModel(weights='DEFAULT')
        else:
            model = SwinModel()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    print(f"Debug: Model '{model.__class__.__name__}' initialized")
    model = model.to(device)
    print("Debug: Model moved to device")
    return model

def initialize_loss(loss_name, model_option = None):
    if loss_name == 'l1':
        criterion = nn.L1Loss()
    elif loss_name == 'l1smooth':
        criterion = nn.SmoothL1Loss()
    elif loss_name == 'l2':
        criterion = nn.MSELoss()
    elif loss_name == 'huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Invalid loss function: {loss_name}")
    return criterion

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)