import torch
from torch import nn 

x = torch.tensor([
    [10],
    [37.78]
],dtype=torch.float32)

y = torch.tensor([
    [50],
    [100]
], dtype=torch.float32)


model = nn.Linear(1,1)
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for i in range(0,100000):

    optimizer.zero_grad()
    ouputs = model(x)
    loss = loss_fn(ouputs,y)
    loss.backward()
    optimizer.step()

    if i % 100 ==0:
        print()
        print("Bias :", model.bias)
        print("Weight :", model.weight)
        print()


measure = torch.tensor([
    [37.5]
], dtype=torch.float32)

model.eval()
with torch.no_grad():
    prediction = model(measure)
    print("Predition : ",prediction )
