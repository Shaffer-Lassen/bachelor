from utils import initialize_model, set_seed
import torch
import time

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = ['scratch', 'dino', 'res50', 'res101', 'res152', 'dense', 'swin']
dummy_input = torch.randn(1, 3, 224, 224).to(device)

for model_name in models:
    model = initialize_model(model_name, device=device)
    total_params, trainable_params = count_parameters(model)
    
    inference_times = []
    for _ in range(1000):
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        end_time = time.time()
        inference_times.append(end_time - start_time)

    average_inference_time = sum(inference_times) / len(inference_times)
    
    print(f"Model: {model_name}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Average Inference Time: {average_inference_time * 1000:.2f} ms")
    print("-" * 50)