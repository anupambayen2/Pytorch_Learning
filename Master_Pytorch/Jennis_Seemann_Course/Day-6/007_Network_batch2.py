import pandas as pd 
import numpy as np 
import torch
from torch import nn 
import sys

df = pd.read_csv('./data/student_exam_data.csv',low_memory=False)

# print(df.head(5))

df.rename(columns={'Study Hours':'study_hours',
                   'Previous Exam Score':'previous_exam_score',
                   'Pass/Fail':'pass_fail'},inplace=True)

X = torch.tensor(df[['study_hours','previous_exam_score']].values,dtype=torch.float32)

y = torch.tensor(df['pass_fail'],dtype=torch.float32).reshape(-1,1)

model = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,1)
)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_entries = X.size(0)
batch_size = 32

for i in range(0,1000):
    loss_sum = 0
    for start in range(0, num_entries, batch_size):
        end = min(num_entries, start + batch_size)
        X_data = X[start:end]
        y_data = y[start:end]

        optimizer.zero_grad()
        outputs = model(X_data)
        loss = loss_fn(outputs,y_data)
        loss.backward()
        loss_sum +=loss.item()
        optimizer.step()

    if i % 10 == 0:
        print(loss_sum)

model.eval()
with torch.no_grad():
    outputs = model(X)
    y_pred = nn.functional.sigmoid(outputs) >0.5
    y_pred_correct = y_pred.type(torch.float32) == y
    print(y_pred_correct.type(torch.float32).mean())
