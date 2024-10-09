import torch
import pytorch_apis

x = torch.randn((100, 20))
y = torch.randn((20, 30))
z = torch.Tensor([0])

xy = torch.mm(x, y)

device = torch.device('cuda')
a = x.to(device)
b = y.to(device)
c = pytorch_apis.gemm(a, b, 100, 30, device)


if xy.shape != c.shape:
    print("WRONG!")
else:
    print('shape are perfectly matched!')
cnt = 0
print(xy)
with torch.no_grad():
    for i in range(xy.shape[0]):
        for j in range(xy.shape[1]):
            if abs(xy[i][j]-c[i][j]) > 1e-4 and xy[i][j] != 0:
                print('xy', xy[i][j])
                print('c', c[i][j])
                cnt += 1


if cnt > 0:
    print('WRONG! cnt > 0')
else:
    print('nums are perfectly matched!')
