import torch 

#create two tensors

a = torch.tensor([1,2,3,4])
b = torch.tensor([10,20,30.40])

print("Tensor a:", a)
print("Tensor b:", b)
print()

#Addition
print("a + b", a+b)
print()

#Subtraction
print("a - b", a-b)
print()

#Mulitiplicatio (Element wise)
print("a * b", a * b)
print()

#Division
print("b/a", b/a)
print()

#Power
print("a**2", a**2)
print()