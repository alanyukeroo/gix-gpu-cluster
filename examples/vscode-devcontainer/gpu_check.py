import torch

print("CUDA available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name())
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok, sum =", (x @ x).sum().item())
