import torch

A = torch.tensor([
    [1,2],
    [2,4]
])

B = torch.tensor([
    [5,6],
    [7,8]
])

# Manual calculation to understand
print("Position [0,0] : (1*5) + (2*7) =", (1*5) + (2*7))
print("Position [0,1] : (1*5) + (2*7) =", (1*6) + (2*8))
print("Position [1,0] : (1*5) + (2*7) =", (2*5) + (4*7))
print("Position [0,0] : (1*5) + (2*7) =", (2*6) + (4*8))
print()

print("Pytorch result:")
print(A @ B)
