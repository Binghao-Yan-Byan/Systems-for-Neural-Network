import torch
import torch.utils
from LeNet300100_Byan import LeNet300100_Byan
from torchvision import datasets, transforms
import time
import datetime
from torch.profiler import profile, record_function, ProfilerActivity

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = LeNet300100_Byan(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    start1 = datetime.datetime.now()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for epoch in range(epochs):
            loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss/len(train_loader):.4f}")
    end1 = datetime.datetime.now()
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))
    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=20))
    #print(prof.key_averages().table(sort_by='self_cuda_memory_usage', row_limit=40))
    difference1 = end1-start1
    print('Total Training time is', difference1)
        

def test(model, test_loader, device):
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
    print(f'Test Accuracy: {accuracy:.2f}%')

train(model, train_loader, criterion, optimizer, device, epochs=5)