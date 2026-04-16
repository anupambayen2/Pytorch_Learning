
import pandas as pd
from torch import nn
import torch
import sys,os

X_mean = torch.load("./model/X_mean.pt", weights_only=True)
X_std = torch.load("./model/X_std.pt", weights_only=True)
y_mean = torch.load("./model/y_mean.pt",weights_only=True)
y_std = torch.load("./model/y_std.pt",weights_only=True)



model = nn.Linear(3,1)

model.load_state_dict(torch.load("./model/model.pt",weights_only=True))

model.eval()

X_test = torch.tensor([
    [10,100000,0],
    [2,5000,0],
    [5,10000,0],
    [3,12000,0]
])

X_test_v2 = torch.tensor([
    [10,100000,1],
    [2,5000,1],
    [5,10000,1],
    [3,12000,1]
])

with torch.no_grad():

    prediction = model((X_test - X_mean)/X_std)
    print(prediction * y_std + y_mean)
    prediction_v1 = model((X_test_v2 - X_mean)/X_std)
    print(prediction_v1 * y_std + y_mean)