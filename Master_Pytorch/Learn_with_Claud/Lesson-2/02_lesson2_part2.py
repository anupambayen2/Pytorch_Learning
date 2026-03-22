import torch 

#Example 1: Simple matrix multiplication

A = torch.tensor([
    [1,2],
    [2,4]
])

B = torch.tensor([
    [5,6],
    [7,8]
])

print("Matrix A:")
print(A)
print("Shape:",A.shape)
print()

print("Matrix B:")
print(B)
print("Shape:",B.shape)
print()


# Element wise multiplication (NOT what we want for neural network)
print("A * B Element-wise:")
print(A * B)
print()

# Matrix multiplication (THIS is what neural network use!)
print("A @ B (Matrix Multiplication):")
print(A @ B)
print()

#Alternative Syntax
print("Torch.matmul(A,B):")
print(torch.matmul(A,B))

# Example 2 : Different shapes
X = torch.tensor([[1,2,3]]) # Shape : [1,3]
W = torch.tensor([
    [4],
    [5],
    [6]
])

print("X Shape:", X.shape)
print("W Shape:", W.shape)

result = X @ W 
print("X @ W result:")
print(result)
print("Result shape:", result.shape)


