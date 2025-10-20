import torch

# 2. Create a random tensor with shape (7, 7)
a = torch.rand((7, 7))
print(a)
print("Shape:", a.shape)

# 3. Perform a matrix multiplication with another random tensor (1, 7)
b = torch.rand((1, 7))
r = torch.matmul(a, b.T)
print(r)

# 4. Set the random seed to 0 and redo steps 2 & 3
torch.manual_seed(0)
a = torch.rand((7, 7))
b = torch.rand((1, 7))
r = a @ b.T
print('a:', a)
print('b:', b)
print('r:', r)

# 5. Set the GPU random seed to 1234
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    print('Set CUDA manual seed to 1234')
else:
    print('CUDA not available on this machine')

# 6. Create two random tensors (2, 3) and send them to GPU if available
d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)
A = torch.rand((2, 3), device=d)
B = torch.rand((2, 3), device=d)
print('A:', A)
print('B:', B)
print('A shape:', A.shape, 'device:', A.device)

# 7. Perform matrix multiplication
R = A @ B.T
print('R:', R)
print('R shape:', R.shape)

# 8. Find max and min values of the output
mx = R.max()
mn = R.min()
print('max value:', mx.item())
print('min value:', mn.item())

# 9. Find max and min index values
amx = R.view(-1).argmax()
amn = R.view(-1).argmin()
amx_2d = (amx // R.shape[1], amx % R.shape[1])
amn_2d = (amn // R.shape[1], amn % R.shape[1])
print('flat argmax index:', amx.item(), 'as 2D index:', amx_2d)
print('flat argmin index:', amn.item(), 'as 2D index:', amn_2d)

# 10. Make random tensor (1, 1, 1, 10) and remove all 1 dimensions
torch.manual_seed(7)
t = torch.rand((1, 1, 1, 10))
print(t, t.shape)
sq = t.squeeze()
print(sq, sq.shape)
