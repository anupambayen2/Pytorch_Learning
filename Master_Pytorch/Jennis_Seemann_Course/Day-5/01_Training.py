import torch 
from torch import nn

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('./data/SMSSpamCollection',sep='\t',names=['type','message'])

df['spam'] = df['type'].map({'spam':1,'ham':0})
df = df.drop('type',axis=1)
cv = CountVectorizer(max_features=1000)
message = cv.fit_transform(df['message'])
# print(cv.get_feature_names_out()[923])

X = torch.tensor(message.todense(),dtype=torch.float32)
y = torch.tensor(df['spam'], dtype=torch.float32).reshape(-1,1)


# print(X.shape)
# print(y.shape)

model = nn.Linear(1000,1)
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)


for i in range(0, 10000):

    # Training Pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs,y)
    loss.backward()
    optimizer.step()

    if i % 100 ==0:
        print(loss)


model.eval()

with torch.no_grad():
    y_pred = nn.functional.sigmoid(model(X)) > 0.4
    accuracy = (y_pred == y).type(torch.float32).mean()
    sensitivity = (y_pred[y==1] == y[y== 1]).type(torch.float32).mean()
    specificity = (y_pred[y==0] == y[y== 0]).type(torch.float32).mean()
    precision = (y_pred[y_pred==1] == y[y_pred== 1]).type(torch.float32).mean()
  
    print('accuracy :', accuracy)
    print('sensitivity :', sensitivity)
    print('specificity :', specificity)
    print('precision :', precision)
   