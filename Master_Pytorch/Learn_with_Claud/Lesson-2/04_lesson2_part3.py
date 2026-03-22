import torch

# create 1D tensor
x = torch.arange(12)
print(x)

print("Shape",x.shape)
print()

#Reshape to 2D (3 rows, 4 columns)
reshape_3x4 = x.reshape(3,4)
print(reshape_3x4)
print(reshape_3x4.shape)

#reshape to 2d (4 rows, 2 columns)

reshape_4x3 = x.reshape(4,3)
print(reshape_4x3)
print(reshape_4x3.shape)

# Use 1 to let Pytorch calculate one dimension
reshape_auto = x.reshape(2,-1)
print("Reshaped to 2x? (auto):")
print(reshape_auto)
print("Shape", reshape_auto.shape)
print()

# Add a dimension (unsqueeze)
unsqueezed = x.unsqueeze(0)
print("Unsqueezed at dim 0:")
print(unsqueezed)
print("Shape", unsqueezed.shape)
print()

# Flatten to 1D

flattened = reshape_3x4.flatten()
print("Flattened")
print(flattened)
print("Shape",flattened.shape)
print()

# reshape the tensor with (2,5)

reshape_2x5 = x.reshape(2,5)
print(reshape_2x5)