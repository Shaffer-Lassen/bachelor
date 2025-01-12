import os
import torch
from torchvision import transforms
from PIL import Image
import argparse
from matplotlib import cm
import numpy as np
from utils import initialize_model, load_checkpoint


def predict_depth(image_path, model, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
    except Exception as e:
        raise RuntimeError(f"Error loading image: {image_path}, Error: {e}")

    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        try:
            depth_prediction = model(input_tensor)
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {e}")

    depth_prediction = depth_prediction.squeeze(0).cpu()
    return depth_prediction, original_size

def save_depth_map(depth_map, save_path, original_size, color_map='spectral'):
    try:
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        if color_map == 'spectral':
            colormap = cm.get_cmap('Spectral_r')
            depth_map_np = depth_map.squeeze().numpy()
            depth_map_colored = colormap(depth_map_np)[:, :, :3]
            depth_map_colored = (depth_map_colored * 255).astype(np.uint8)
            depth_image = Image.fromarray(depth_map_colored)
        elif color_map == 'grayscale':
            depth_map = (depth_map * 255).to(torch.uint8)
            depth_image = transforms.ToPILImage()(depth_map.squeeze())
        else:
            raise ValueError(f"Invalid color map option: {color_map}")

        depth_image = depth_image.resize(original_size, Image.BICUBIC)
        depth_image.save(save_path)
    except Exception as e:
        raise RuntimeError(f"Error saving depth map to {save_path}. Original error: {e}")



def save_side_by_side(original_image, depth_map, save_path, color_map='spectral'):
    try:
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        if color_map == 'spectral':
            colormap = cm.get_cmap('Spectral')
            depth_map_np = depth_map.squeeze().numpy()
            depth_map_colored = colormap(depth_map_np)[:, :, :3]
            depth_map_colored = (depth_map_colored * 255).astype(np.uint8)
            depth_image = Image.fromarray(depth_map_colored)
        elif color_map == 'grayscale':
            depth_map = (depth_map * 255).to(torch.uint8)
            depth_image = transforms.ToPILImage()(depth_map.squeeze())
        else:
            raise ValueError(f"Invalid color map option: {color_map}")

        depth_image = depth_image.resize(original_image.size, Image.BICUBIC)
        combined_width = original_image.width + depth_image.width
        combined_image = Image.new("RGB", (combined_width, original_image.height))
        combined_image.paste(original_image, (0, 0))
        combined_image.paste(depth_image, (original_image.width, 0))
        combined_image.save(save_path)
    except Exception as e:
        raise RuntimeError(f"Error saving side-by-side image to {save_path}. Original error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Make depth prediction of 1 or more images')
    parser.add_argument('--image_path', type=str, default='data', help='Path to input image or directory of images')
    parser.add_argument('--save_dir', type=str, default='save', help='Directory to save predicted depth maps')
    parser.add_argument('--model', type=str, default='scratch', help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the saved model checkpoint')
    parser.add_argument('--color_map', type=str, default='grayscale', choices=['spectral', 'grayscale'], help='Color map to use for depth maps')
    parser.add_argument('--side_by_side', action='store_true', help='Save original image and depth map side-by-side')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = initialize_model(args.model, device)
    model = load_checkpoint(model=model, checkpoint_path=args.checkpoint, device=device)

    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if os.path.isdir(args.image_path):
        image_paths = [os.path.join(args.image_path, fname) for fname in os.listdir(args.image_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    else:
        image_paths = [args.image_path]

    os.makedirs(args.save_dir, exist_ok=True)

    for image_path in image_paths:
        try:
            original_image = Image.open(image_path).convert('RGB')
            depth_map, original_size = predict_depth(image_path, model, transform, device)
            
            save_name = os.path.splitext(os.path.basename(image_path))[0] + f"_{args.model}" + ('_side_by_side.png' if args.side_by_side else '_depth.png')
            save_path = os.path.join(args.save_dir, save_name)
            
            if args.side_by_side:
                save_side_by_side(original_image, depth_map, save_path, args.color_map)
            else:
                save_depth_map(depth_map, save_path, original_size, args.color_map)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")


if __name__ == '__main__':
    main()
