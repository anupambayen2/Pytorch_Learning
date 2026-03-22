import torch

#Create inputs
x = torch.tensor([2.0],requires_grad=True)
w = torch.tensor([3.1],requires_grad=True)
b = torch.tensor([1.0],requires_grad=True)

print("x:",x)
print("w:",w)
print("b:",b)
print()

# Complex equation = y = w * x = b
y = w * x + b
print("y = w * x +b", y)
print()

#Compute gradients
y.backward()

print("Gradient dy/dx:", x.grad) 
print("Gradient dy/dw:", w.grad)
print("Gradient dy/db:", b.grad)
print()

# Manual verification:

print("Manual Calculation:")
print("y =3 * 2+1 =", 3 * 2 + 1)
print("dy/dx", 3.0)
print("dy/dw", 2.0)
print("dy/db", 1.0)



