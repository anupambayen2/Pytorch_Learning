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

hidden_model = nn.Linear(2,10)
output_model = nn.Linear(10,1)
loss_fn = torch.nn.BCEWithLogitsLoss()
parameters = list(hidden_model.parameters()) + list(output_model.parameters())
optimizer = torch.optim.SGD(parameters, lr=0.005)



for i in range(0,300000):
    optimizer.zero_grad()
    outputs = hidden_model(X)
    outputs = nn.functional.sigmoid(outputs)
    outputs = output_model(outputs)
    loss = loss_fn(outputs,y)
    loss.backward()
    optimizer.step()

    if i % 10000 == 0:
        print(loss)



hidden_model.eval()
output_model.eval()
with torch.no_grad():
    outputs = hidden_model(X)
    outputs = nn.functional.sigmoid(outputs)
    outputs = output_model(outputs)

    y_pred = nn.functional.sigmoid(outputs) >0.5
    y_pred_correct = y_pred.type(torch.float32) == y
    print(y_pred_correct.type(torch.float32).mean())
