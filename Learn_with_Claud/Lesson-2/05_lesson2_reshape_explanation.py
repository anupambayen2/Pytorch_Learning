import torch

x = torch.arange(12) # 12 elements total

print('Original',x)
print()

print("Total elements:", x.numel())
print()

#Example-1: Reshape(2,-1)
#Logic : elements /2 rows = 6 columns
result1 = x.reshape(2,-1)
print(result1)
print("Pytorch calculated : 12 /2 =6 ")
print()

#Example-2: Reshape(-1,3)
#Logic 12 elements /3 columns = 4 rows
result2 = x.reshape(-1,3)
print(result2)
print()

#Example 3: reshape(3,-1)
#Logic: 12 elements /3 rows = 4 columns
result3 = x.reshape(3,-1)
print(result3)


## Practice questions
print()
y = torch.arange(20)

reshape_1 = y.reshape(4,-1)
print(reshape_1)
print(reshape_1.shape)

print()
z = torch.arange(15)
reshape_2 = z.reshape(-1,5)
print(reshape_2)
print(reshape_2.shape)

print()
xx = torch.arange(18)
print(xx)
reshape_3 = xx.reshape(4,-1)
print(reshape_3)
print()