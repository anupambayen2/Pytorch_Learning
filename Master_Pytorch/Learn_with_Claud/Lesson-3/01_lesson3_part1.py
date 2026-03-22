import torch 

#create a tensor and tell PyTorch to track operation on it
x = torch.tensor([3.0], requires_grad=True) # requires_grad=True is the key
print("X", x)
print("Does x require gradients?", x.requires_grad)
print()

#Do some math
y = x ** 2 # y = x2
print("Y = x2",y)
print()

#Compute gradient (derivative)
y.backward() # This calculates dy/dx

# Check the gradient
print("Gradient dy/dx", x.grad)
print()

# Manual calculation to verify
derivatives = 2 * 3
print(derivatives)
print()