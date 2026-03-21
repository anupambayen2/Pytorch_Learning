import torch

#Create a 2D tensor

matrix = torch.tensor([
    [10,20,30],
    [40,50,60],
    [70,80,90]
])

print("Original Matrix:")
print(matrix)
print()

# Indexing (Accessing elements)
print("Element at row 0, column 0:", matrix[0,0]) #First Element
print("Element at row 1, column 2:", matrix[1,2]) #Second row, third column
print()

# Slicing (Accessing rows/columns)
print("First row:", matrix[0,:]) # row 0, all columns
print("Last column:", matrix[:,2]) # All rows, column 2
print("First 2 rows", matrix[:2,:]) # First 2 rows, all columns
print()

# Modifying values
matrix[0,0] = 999
print("After chaning the first element")
print(matrix)
print()

# Fancy Indexing
print("Top Left 2X2 blocks :", matrix[:2,:2])

print("Accessing :",matrix[2,1])

print("2nd Rosw and all colums:",matrix[1:2])


#Method -1
print("Method-1", matrix[1:2])

#Method -2
print("Method-2", matrix[1])

#Method -3
print("Method-3", matrix[1,:])