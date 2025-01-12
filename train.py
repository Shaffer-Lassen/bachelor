import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
from tqdm import tqdm
from datetime import datetime

from data import load_nyu_train, load_coco_train
from transform import PrepareDepthTransform, PrepareImageTransform, PairedTransform
from utils import initialize_model, initialize_loss, set_seed, seed_worker
from metrics import rmse, threshold_accuracy, log10_error, abs_rel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NYUv2', choices=['NYUv2', 'COCO'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--model', type=str, default='dino')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--loss', type=str, default='l1')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--wandb', action='store_true', help="Enable WandB logging (default: False)")
    args = parser.parse_args()
    
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    set_seed(1)
    
    if args.wandb:
        wandb.init(project="depth-estimation", config={
            "model": args.model,
            "Loss": args.loss,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "data": args.dataset
        })
    else:
        print("WandB logging is disabled.")

    transform_pair = PairedTransform()
    transform_image = PrepareImageTransform()
    transform_depth = PrepareDepthTransform()
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(model_name=args.model, device=device, pretrained=args.pretrained)

    print(f"Model '{model.__class__.__name__}' was created")

    if args.dataset == 'NYUv2':
        train_subset, val_subset = load_nyu_train(transform_image=transform_image, transform_depth=transform_depth, transform_pair=transform_pair)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        
    elif args.dataset == 'COCO':
        train_subset, val_subset = load_coco_train(transform_image=transform_image, transform_depth=transform_depth, transform_pair=transform_pair)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        raise ValueError('Not a valid dataset name')
    
    print(f"Training set {args.dataset} of size: {len(train_subset)} loaded")
    print(f"Validation set {args.dataset} of size: {len(val_subset)} loaded")

    criterion = initialize_loss(args.loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    print("Training initiated")
    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        print(f"Epoch {epoch + 1}/{args.epochs}")
        with tqdm(total=len(train_loader), desc="Training Progress", unit="batch") as pbar:
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                
                if i % 800 == 0:
                    mask = labels >= 0 
                    min_label = torch.min(labels[mask]).item()
                    max_label = torch.max(labels[mask]).item()
                    avg_label = torch.mean(labels[mask]).item()
                    
                    print(f"Labels - Min: {min_label:.4f}, Max: {max_label:.4f}, Avg: {avg_label:.4f}")  
                    
                    min_output = torch.min(outputs).item()
                    max_output = torch.max(outputs).item()
                    avg_output = torch.mean(outputs).item()
                    print(f"Outputs - Min: {min_output:.4f}, Max: {max_output:.4f}, Avg: {avg_output:.4f}")
            
                optimizer.zero_grad()
                mask = labels >= 0
                masked_outputs = outputs * mask
                masked_labels = labels * mask
                loss = criterion(masked_outputs, masked_labels)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                pbar.update(1)

        average_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        rmse_running = 0
        abs_rel_running = 0
        log10_running = 0
        delta_1_running = 0
        delta_2_running = 0
        delta_3_running = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="Validation Progress", unit="batch") as vbar:
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    mask = labels >= 0
                    outputs_masked = outputs*mask
                    labels_masked = labels*mask
                    if i % 100 == 0:
                        min_label = torch.min(labels).item()
                        max_label = torch.max(labels).item()
                        avg_label = torch.mean(labels).item() 
                        print(f"Labels - Min: {min_label:.4f}, Max: {max_label:.4f}, Avg: {avg_label:.4f}")         
                        min_output = torch.min(outputs).item()
                        max_output = torch.max(outputs).item()
                        avg_output = torch.mean(outputs).item()
                        print(f"Outputs - Min: {min_output:.4f}, Max: {max_output:.4f}, Avg: {avg_output:.4f}")

                    loss = criterion(outputs_masked, labels_masked)
                    val_loss += loss.item()
                    
                    outputs= outputs[mask]
                    labels = labels[mask]
                    rmse_running += rmse(outputs, labels).item()
                    abs_rel_running += abs_rel(outputs, labels).item()
                    log10_running += log10_error(outputs, labels).item()
                    delta_1_running += threshold_accuracy(outputs, labels, threshold=1.25).item()
                    delta_2_running += threshold_accuracy(outputs, labels, threshold=1.25**2).item()
                    delta_3_running += threshold_accuracy(outputs, labels, threshold=1.25**3).item()

                    vbar.update(1)

        average_val_loss = val_loss / len(val_loader)
        average_rmse = rmse_running / len(val_loader)
        average_abs_rel = abs_rel_running / len(val_loader)
        average_log10 = log10_running / len(val_loader)
        average_delta_1 = delta_1_running / len(val_loader)
        average_delta_2 = delta_2_running / len(val_loader)
        average_delta_3 = delta_3_running / len(val_loader)

        print(f"Epoch {epoch + 1} Validation Metrics:")
        print(f"  Loss: {average_val_loss:.4f}")
        print(f"  RMSE: {average_rmse:.4f}")
        print(f"  AbsRel: {average_abs_rel:.4f}")
        print(f"  Log10: {average_log10:.4f}")
        print(f"  Delta < 1.25: {average_delta_1:.4f}")
        print(f"  Delta < 1.25^2: {average_delta_2:.4f}")
        print(f"  Delta < 1.25^3: {average_delta_3:.4f}")

        if args.wandb:
            wandb.log({
            "train_loss": average_train_loss,
            "val_loss": average_val_loss,
            "RMSE": average_rmse,
            "AbsRel": average_abs_rel,
            "Log10": average_log10,
            "Delta < 1.25": average_delta_1,
            "Delta < 1.25^2": average_delta_2,
            "Delta < 1.25^3": average_delta_3,
            "epoch": epoch + 1
            })

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_filename = f"{timestamp}_{args.model}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': average_train_loss,
                'val_loss': average_val_loss,
                'loss': criterion,
            }, os.path.join(checkpoint_dir, checkpoint_filename))
            print(f"Checkpoint saved at epoch {epoch + 1} as '{checkpoint_filename}'")

if __name__ == '__main__':
    main()
