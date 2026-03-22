import torch
 
# 1. Create two tensors:

#    - x: shape [3, 4] with random values

#    - y: shape [4] with values [1, 2, 3, 4]
 
# 2. Add x and y (broadcasting will happen!)
 
# 3. Create a weight matrix W of shape [4, 2] with random values
 
# 4. Multiply x @ W (matrix multiplication)
 
# 5. Reshape the result to shape [1, 6]
 
# 6. Print all shapes and results

x = torch.arange(12,24)
print(x.shape)
x_reshape = x.reshape(3,4)
print(x_reshape)
print(x_reshape.shape)

y = torch.tensor([1,2,3,4])
print(y)

result = x_reshape + y
print(result)
print()

ww = torch.arange(8,16)
print(ww)
w = ww.reshape(4,2)
print("X Reshape",x_reshape)
print("Weight",w)
print(w.shape)

result2 = x_reshape @ w
print("Multiplication", result2)
print()
result3 = result2.reshape(1,6)
print("Reshape Result", result3)
print()
 