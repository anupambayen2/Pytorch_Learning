import torch 
from torch import nn 

print(type(5+5.4))


x = torch.tensor([
    [10],
    [38],
    [100],
    [150]
],dtype=torch.float32)

print(x)
print()
result = x * 0.5
x = x.type(torch.int64)
print(x)

print(result.dtype)
print()
print(result)
print()

print((10*1.8)+32)