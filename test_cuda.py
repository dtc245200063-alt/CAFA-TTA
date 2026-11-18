
import torch
import torch.nn as nn

# 1. Kiểm tra CUDA
assert torch.cuda.is_available(), "CUDA is not available"

device = torch.device("cuda")

print("=" * 50)
print("CAFA-TTA | CUDA TRAINING TEST")
print("=" * 50)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))

# 2. Tạo mạng neural đơn giản
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
).to(device)

# 3. Tạo dữ liệu giả lập trên GPU
x = torch.randn(256, 128, device=device)
y = torch.randint(0, 2, (256,), device=device)

# 4. Loss Function và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

# 5. Training Loop
model.train()

for epoch in range(1, 11):
    optimizer.zero_grad()

    outputs = model(x)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    print(
        f"Epoch {epoch:02d}/10 "
        f"| Loss: {loss.item():.4f}"
    )

# 6. Thống kê VRAM
torch.cuda.synchronize()

allocated = torch.cuda.memory_allocated() / 1024**2
peak = torch.cuda.max_memory_allocated() / 1024**2

print("=" * 50)
print("TRAINING COMPLETED")
print(f"GPU Memory Allocated: {allocated:.2f} MB")
print(f"Peak GPU Memory: {peak:.2f} MB")
print("Model Device:", next(model.parameters()).device)
print("=" * 50)
