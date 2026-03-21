import torch 

#Example 1 -- Create a 1D tensor
tensor_1d = torch.tensor([1,2,3,4,5])

print('1D Tensor:', tensor_1d)
print("Shape:", tensor_1d.shape)
print("Data Type:", tensor_1d.dtype)
print()

#Example 2 -- Create a 2D tensor (matrix)
tensor_2d = torch.tensor([
    [1,2,3],
    [4,5,6]
])

print('2D Tensor:', tensor_2d)
print("Shape:", tensor_2d.shape)
print("Data Type:", tensor_2d.dtype)
print()

#Example 3 : Create a 3D tensor
tensor_3d = torch.tensor([
    [[1,2],[3,4]],
    [[5,6],[7,8]]
])

print('3D Tensor:', tensor_3d)
print("Shape:", tensor_3d.shape)
print("Data Type:", tensor_3d.dtype)
print()





