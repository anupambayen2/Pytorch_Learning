import pandas as pd
from torch import nn
import torch
import sys,os

car_data = pd.read_csv(r'/Users/anupambayen/projects/deep_learning/Master_Pytorch/Jennis_Seemann_Course/Day-3-4/Car_Price/data/used_cars.csv', low_memory=False)

price = car_data['price']
price = price.str.replace('$','')
price = price.str.replace(',','')
price = price.astype(int)

car_data['accident_flag'] = car_data['accident'].apply(
    lambda x: 1 if str(x).strip().lower() == 'at least 1 accident or damage reported' else 0
)



age = car_data['model_year'].max() - car_data['model_year']

milage = car_data['milage']
milage = milage.str.replace(',','')
milage = milage.str.replace(' mi.','')
milage = milage.astype(int)

accident = car_data['accident_flag']
accident = accident.astype(int)

if not os.path.isdir("./model"):
    os.mkdir("./model")

# Torch : creating X and y data as tensors

X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32),
    torch.tensor(accident,dtype=torch.float32)
])
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X = (X-X_mean)/X_std

torch.save(X_mean,"./model/X_mean.pt")
torch.save(X_std,"./model/X_std.pt")

# sys.exit()

y = torch.tensor(price,dtype=torch.float32).reshape((-1,1))

y_mean = y.mean()
y_std = y.std()
y = (y- y_mean)/y_std


torch.save(y_mean,"./model/y_mean.pt")
torch.save(y_std,"./model/y_std.pt")


model = nn.Linear(3,1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(0,1000):
    #Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs,y)
    loss.backward()
    optimizer.step()
    # print(loss)

  
torch.save(model.state_dict(),"./model/model.pt")


