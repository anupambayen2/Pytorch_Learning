import torch

# b = torch.tensor(32)
# w1 = torch.tensor(1.8)

# x1 = torch.tensor([10,38,100,150])

# y_pred = 1 * b + x1 * w1 

# print(y_pred[1].item())
# print(b.shape)
# print(x1.shape)

# print(b.size())
# print(x1.size())

#matrix in tensor
x = torch.tensor([
    [10],
    [12],
    [38],
    [34],

])

print(x)
print(x.shape)
print(x.size(1))
print(x[:,0])
