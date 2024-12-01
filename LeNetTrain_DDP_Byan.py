import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import torch.optim as optim
from LeNet_Byan import LeNet_Byan
from torchvision import datasets, transforms
    
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def ddp_train(rank, world_size):
    setup_ddp(rank, world_size)


    device = torch.device(f'cuda:{rank}')
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

    model = LeNet_Byan(device)
    model = DDP(model, device_ids=[rank])
    epochs = 20
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss += loss.item()
        print(f"Rank {rank}, Epoch {epoch + 1}/{epochs}, Loss: {loss/len(train_loader):.4f}")
        test(rank, model, test_loader, device)
    cleanup_ddp()

def test(rank, model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100*correct/total
    print(f'Rank {rank}, Test Accuracy: {accuracy:.2f}%')

def main():
    world_size = 2
    mp.spawn(ddp_train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()