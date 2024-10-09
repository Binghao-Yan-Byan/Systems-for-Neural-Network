import torch
import pytorch_apis
# Create two matrices with requires_grad=True to track operations
A = torch.randn(1, 1000, requires_grad=True)
B = torch.randn(1000, 1, requires_grad=True)


D = A.clone().detach().requires_grad_(True)
E = B.clone().detach().requires_grad_(True)
F = torch.mm(D, E)
device = torch.device('cuda')
A_ = A.to(device)
B_ = B.to(device)
C = pytorch_apis.gemm(A_, B_, 1, 1, device)
# Perform matrix multiplication using torch.mm

# Perform some further operations (e.g., summing elements)

output = C.sum()

output2 = F.sum()

# Perform backpropagation (compute gradients)
output.backward()

output2.backward()

print(abs(output.item()-output2.item())<1e-4)
# Display gradients
with torch.no_grad():
    cnt = 0

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A.grad[i][j].item() - D.grad[i][j].item()) > 1e-4:
                cnt += 1
    if cnt==0:
        print('Perfect Math Gradient A')
    else:
        print("WRONG!")

    cnt = 0
    with torch.no_grad():
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if abs(B.grad[i][j] - E.grad[i][j])> 1e-4:
                    cnt += 1
    if cnt==0:
        print('Perfect Math Gradient B')
    else:
        print("WRONG!")
