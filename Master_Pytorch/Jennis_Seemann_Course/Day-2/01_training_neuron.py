import torch 
from torch import nn

x1 = torch.tensor([[10]], dtype=torch.float32) # temp in C
y1 = torch.tensor([[50]], dtype=torch.float32) # temp in f

x2 = torch.tensor([[37.78]], dtype=torch.float32) # temp in c
y2 = torch.tensor([[100.0]], dtype=torch.float32) # temp in f

model = nn.Linear(1,1)
loss_fn = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)



# Trainning Pass
for i in range(0,100000):
    optimizer.zero_grad()
    outputs = model(x1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    outputs = model(x2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()

    if i % 100 ==0:
        print("Weight - ",model.weight)
        print("Bias - ",model.bias)
        print()

print()



y1_pred = model(x1)

print("y_pred=", y1_pred)