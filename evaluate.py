#!/usr/bin/env python

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from transform import PrepareImageTransform, PrepareDepthTransform, PairedTransform
from data import load_nyu_test
from utils import initialize_model, set_seed


class DepthMetrics:
    @staticmethod
    def create_mask(gt, pred=None, min_depth=1e-3):
        mask = (gt > min_depth) 
        if pred is not None:
            mask = mask & (pred > min_depth)
        return mask

    @staticmethod
    def abs_rel(pred, gt):
        mask = DepthMetrics.create_mask(gt, pred)
        pred = pred[mask]
        gt = gt[mask]
        if gt.numel() == 0:
            return torch.tensor(0.0).to(pred.device)
        return torch.mean(torch.abs(gt - pred) / gt)

    @staticmethod
    def rmse(pred, gt):
        mask = DepthMetrics.create_mask(gt, pred)
        pred = pred[mask]
        gt = gt[mask]
        if gt.numel() == 0:
            return torch.tensor(0.0).to(pred.device)
        return torch.sqrt(torch.mean((gt - pred) ** 2))

    @staticmethod
    def log10_error(pred, gt):
        mask = DepthMetrics.create_mask(gt, pred)
        pred = pred[mask]
        gt = gt[mask]
        if gt.numel() == 0:
            return torch.tensor(0.0).to(pred.device)
        return torch.mean(torch.abs(torch.log10(pred) - torch.log10(gt)))

    @staticmethod
    def threshold_accuracy(pred, gt, threshold=1.25):
        mask = DepthMetrics.create_mask(gt, pred)
        pred = pred[mask]
        gt = gt[mask]
        if gt.numel() == 0:
            return torch.tensor(0.0).to(pred.device)
        ratio = torch.maximum(pred / gt, gt / pred)
        return torch.mean((ratio < threshold).float())

    @staticmethod
    def scale_invariant_error(pred, gt):
        mask = DepthMetrics.create_mask(gt, pred)
        pred = pred[mask]
        gt = gt[mask]
        if gt.numel() == 0:
            return torch.tensor(0.0).to(pred.device)
        log_diff = torch.log(pred) - torch.log(gt)
        return torch.sqrt(torch.mean(log_diff ** 2) - 0.5 * (torch.mean(log_diff) ** 2))

def save_depth_visualization(pred, gt, rgb, save_dir, index):
    os.makedirs(save_dir, exist_ok=True)
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    rgb = rgb.cpu().numpy().transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    error_map = np.abs(gt - pred)
    error_map = error_map / error_map.max()
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes[0, 0].imshow(rgb)
    axes[0, 0].axis('off')
    pred_plot = axes[0, 1].imshow(pred, cmap='plasma')
    axes[0, 1].axis('off')
    plt.colorbar(pred_plot, ax=axes[0, 1])
    gt_plot = axes[1, 0].imshow(gt, cmap='plasma')
    axes[1, 0].axis('off')
    plt.colorbar(gt_plot, ax=axes[1, 0])
    error_plot = axes[1, 1].imshow(error_map, cmap='hot')
    axes[1, 1].axis('off')
    plt.colorbar(error_plot, ax=axes[1, 1])
    plt.savefig(os.path.join(save_dir, f'depth_viz_{index}.png'))
    plt.close()

def evaluate(model, test_loader, device, save_viz=True):
    model.eval()
    metrics = {
        "AbsRel": 0, "RMSE": 0, "Log10": 0,
        "A1": 0, "A2": 0, "A3": 0,
        "SIE": 0
    }
    n_samples = 0
    viz_dir = './evaluation_results'
    os.makedirs(viz_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (rgb, depth) in enumerate(tqdm(test_loader, desc="Evaluating")):
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            pred_depth = model(rgb)
            pred_depth = F.interpolate(pred_depth, size=depth.shape[-2:],
                                     mode='bilinear', align_corners=False)
            if i == 0:
                print("\nDepth ranges:")
                print(f"Ground Truth shape: {depth.shape}")
                print(f"Prediction shape: {pred_depth.shape}")
                print(f"Ground Truth - Min: {depth.min():.3f}, Max: {depth.max():.3f}")
                print(f"Prediction  - Min: {pred_depth.min():.3f}, Max: {pred_depth.max():.3f}\n")
                print(f"Prediction  - Avg: {pred_depth.mean():.3f}, Ground truth Avg: {depth.mean():.3f}\n")
            
            metrics["AbsRel"] += DepthMetrics.abs_rel(pred_depth, depth).item()
            metrics["RMSE"] += DepthMetrics.rmse(pred_depth, depth).item()
            metrics["Log10"] += DepthMetrics.log10_error(pred_depth, depth).item()
            metrics["A1"] += DepthMetrics.threshold_accuracy(pred_depth, depth, 1.25).item()
            metrics["A2"] += DepthMetrics.threshold_accuracy(pred_depth, depth, 1.25**2).item()
            metrics["A3"] += DepthMetrics.threshold_accuracy(pred_depth, depth, 1.25**3).item()
            metrics["SIE"] += DepthMetrics.scale_invariant_error(pred_depth, depth).item()
            
            if save_viz and i < 5:
                save_depth_visualization(
                    pred_depth[0, 0], depth[0, 0], rgb[0],
                    viz_dir, i
                )
            
            n_samples += 1
    
    for key in metrics:
        metrics[key] /= n_samples
    
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"{'Metric':>10} | {'Value':>10}")
    print("-" * 80)
    for key, value in metrics.items():
        print(f"{key:>10} | {value:>10.4f}")
    print("-" * 80)
    
    return metrics

def main():
    set_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, default='scratch')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_viz', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = initialize_model(args.model, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully.")

    transform_image = PrepareImageTransform()
    transform_depth = PrepareDepthTransform()
    transform_pair = PairedTransform()

    test_data = load_nyu_test(transform_depth=transform_depth, transform_image=transform_image, transform_pair=transform_pair)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    _ = evaluate(model, test_loader, device, save_viz=args.save_viz)

if __name__ == '__main__':
    main()