import torch

print("=== GPU CHECK ===")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Current Device: {torch.cuda.current_device()}")
else:
    print("No CUDA GPU detected")

# Test tensor creation
print("\n=== TENSOR TEST ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create test tensor
x = torch.randn(1000, 1000).to(device)
y = torch.mm(x, x)
print(f"Test computation successful on {device}")

print("\nGPU Status: READY" if torch.cuda.is_available() else "GPU Status: NOT AVAILABLE") 