import torch
import math
import torch.nn.init as init
import pytorch_apis

class Linear_Byan(torch.nn.Module):
    # Do similar things like torch.nn.modules.Linear
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return pytorch_apis.gemm(input, self.weight, input.shape[0], self.out_features, self.device) + self.bias
    



if __name__ == "__main__":
    device = torch.device('cuda')
    x = torch.randn(10, 10).to(device)
    linear = Linear_Byan(10, 2, bias=True, device=device)
    for epoch in range(10):
        y = linear(x)
        criterion = torch.nn.MSELoss()
        optim = torch.optim.SGD(linear.parameters(), lr=0.01)

        optim.zero_grad()
        loss = criterion(y, torch.randn(10, 2).to(torch.device('cuda')))
        print(epoch, loss.item())
        loss.backward()
        optim.step()
        



    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    with torch.no_grad():
        print(linear.bias.grad.data)
        print(linear.weight.grad.data)
        print(linear.bias.data)
        print(linear.weight.data)
    optim.step()
    with torch.no_grad():
        print(linear.bias.data)
        print(linear.weight.data)

    optim.zero_grad()
    loss = criterion(y, torch.randn(10, 2).to(torch.device('cuda')))
    loss.backward()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    with torch.no_grad():
        print(linear.bias.grad.data)
        print(linear.weight.grad.data)
        print(linear.bias.data)
        print(linear.weight.data)
    optim.step()
    with torch.no_grad():
        print(linear.bias.data)
        print(linear.weight.data)

    optim.zero_grad()
    loss = criterion(y, torch.randn(10, 2).to(torch.device('cuda')))
    loss.backward()

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    with torch.no_grad():
        print(linear.bias.grad.data)
        print(linear.weight.grad.data)
        print(linear.bias.data)
        print(linear.weight.data)
