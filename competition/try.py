import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("✅ Using device:", device)

if device.type == 'cuda':
    print("🚀 CUDA is available")
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ Running on CPU")
