import torch
from Linear_Byan import Linear_Byan

class LeNet300100_Byan(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.fc1 = Linear_Byan(28*28, 300, device=device)
        self.fc2 = Linear_Byan(300, 100, device=device)
        self.fc3 = Linear_Byan(100, 10, device=device)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x