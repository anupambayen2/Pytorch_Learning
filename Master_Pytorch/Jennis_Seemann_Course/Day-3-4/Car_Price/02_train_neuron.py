import pandas as pd
from torch import nn
import torch
import sys

car_data = pd.read_csv(r'/Users/anupambayen/projects/deep_learning/Master_Pytorch/Jennis_Seemann_Course/Day-3-4/Car_Price/data/used_cars.csv', low_memory=False)

price = car_data['price']
price = price.str.replace('$','')
price = price.str.replace(',','')
price = price.astype(int)



age = car_data['model_year'].max() - car_data['model_year']

milage = car_data['milage']
milage = milage.str.replace(',','')
milage = milage.str.replace(' mi.','')
milage = milage.astype(int)

# print(price)
# print(age)
# print(milage)

# Torch : creating X and y data as tensors

X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X = (X-X_mean)/X_std
# print(X)

# sys.exit()

y = torch.tensor(price,dtype=torch.float32).reshape((-1,1))

y_mean = y.mean()
y_std = y.std()
y = (y- y_mean)/y_std

# sys.exit()


model = nn.Linear(2,1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(0,1000):
    #Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs,y)
    loss.backward()
    optimizer.step()

    if i % 100 ==0:
        print(loss)

    # if i % 100 ==0:
    #     print(model.bias)
    #     print(model.weight)


X_test = torch.tensor([
    [10,100000]
])


prediction = model((X_test - X_mean)/X_std)
print(prediction * y_std + y_mean)
