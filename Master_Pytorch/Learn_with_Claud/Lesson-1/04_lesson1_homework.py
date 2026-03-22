# 1. Create a 4x5 tensor filled with random numbers between 0 and 1 
# 2. Print its shape 
# 3. Extract the second row 
# 4. Extract the last column # 
# 5. Change all values in the first row to 0 
# 6. Print the final tensor

import torch

matrix = torch.rand(4,5)

print("4*5 matrix")
print(matrix)
print()
print("Shape of Tensor", matrix.shape)
print()
print("Exract Second Row", matrix[1,])
print()
print("Extact last column", matrix[:,4])
print()

matrix[0,]=0
print(matrix)
