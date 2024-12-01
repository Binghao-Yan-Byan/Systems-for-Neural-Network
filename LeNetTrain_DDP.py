
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from LeNet import LeNet300100

# Setup and cleanup functions for DDP
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

# Training and evaluation functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Main DDP function
def ddp_main(rank, world_size):
    setup_ddp(rank, world_size)

    # Configure device
    device = torch.device(f'cuda:{rank}')

    # Dataset and distributed sampler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Model, loss function, and optimizer
    model = LeNet300100().to(device)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Rank {rank}, Epoch {epoch + 1}, Loss: {train_loss:.4f}")

    cleanup_ddp()

# Main entry point
def main():
    world_size = 2  # Number of GPUs
    mp.spawn(ddp_main, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
