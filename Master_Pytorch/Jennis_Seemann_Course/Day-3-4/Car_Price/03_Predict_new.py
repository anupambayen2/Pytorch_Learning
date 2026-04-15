
import pandas as pd
from torch import nn
import torch
import sys,os

X_mean = torch.load("./model/X_mean.pt", weights_only=True)
X_std = torch.load("./model/X_std.pt", weights_only=True)
y_mean = torch.load("./model/y_mean.pt",weights_only=True)
y_std = torch.load("./model/y_std.pt",weights_only=True)



model = nn.Linear(2,1)

model.load_state_dict(torch.load("./model/model.pt",weights_only=True))

model.eval()

X_test = torch.tensor([
    [10,100000],
    [2,5000],
    [5,10000]
])

with torch.no_grad():

    prediction = model((X_test - X_mean)/X_std)
    print(prediction * y_std + y_mean)