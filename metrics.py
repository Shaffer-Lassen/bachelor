import torch
import torch.nn as nn
import torch.nn.functional as F
    
def rmse(predictions, targets):
    if predictions.device != targets.device:
        targets = targets.to(predictions.device)
    
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse)

def abs_rel(pred, target):
    target = torch.clamp(target, min=1e-5)
    return torch.mean(torch.abs(pred - target) / target)


def log10_error(pred, target):
    pred = torch.clamp(pred, min=1e-5)
    target = torch.clamp(target, min=1e-5)
    return torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))

def threshold_accuracy(pred, target, threshold=1.25):
    pred = torch.clamp(pred, min=1e-5)
    target = torch.clamp(target, min=1e-5)
    ratio = torch.max(pred / target, target / pred)
    return torch.mean((ratio < threshold).float())
