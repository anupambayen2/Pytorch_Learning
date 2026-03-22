import torch

#Example 1: Add a scaler to a tensor
x = torch.tensor([1,2,3,4])
print("x:", x)
print("x + 10:", x+10) #Add 10 to every elements
print()

#Example 2: Add tensors with different shapes
a = torch.tensor([
    [1,2,3],
    [4,5,6] # shape 2,3
])

b = torch.tensor([10,20,30])
print(a)
print()

print(b)
print()

result = a + b
print(result)
print()

#column wise broadcasting
c = torch.tensor([
    [100],
    [200] # shape 2,1
])

result2 = a + c
print(result2)
print()

b_broadcasted = b.unsqueeze(0).expand(2,3)
print(b_broadcasted)
print()

