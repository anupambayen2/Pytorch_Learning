import torch

# Method 1 : Tensor filled with Zeroes
zeroes = torch.zeros(3,4) # 3 rows 4 colums
print("Zeroes Tensor")
print(zeroes)
print("Shape", zeroes.shape)
print()

# Method 2 : Tensor filled with Ones
ones = torch.ones(2,3)
print("One Tensor")
print(ones)
print()

# Method 3 : Random numbers (0 to 1)
random = torch.rand(2,2)
print("Random Tensor")
print(random)
print()

# Method 4 : Random Integers
random_int = torch.randint(0,10,(3,3)) # Random integers from 0 to 9
print("Random integers (0-9):")
print(random_int)
print()

# Method 5 : Create a range of numbers
range_tensor = torch.arange(0,10,2) # start, End, Step
print("Range Tensor (0 to 10, step2)")
print(range_tensor)
print()

# Method 6 : Tensor with specific data types(float32)
float_tensor = torch.tensor([1,2,3], dtype=torch.float32)
print("Float Tensor")
print(float_tensor)
print("Data Type :", float_tensor.dtype)