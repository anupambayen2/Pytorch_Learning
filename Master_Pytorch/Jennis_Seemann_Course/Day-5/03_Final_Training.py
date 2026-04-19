import torch 
from torch import nn

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('./data/SMSSpamCollection',sep='\t',names=['type','message'])

df['spam'] = df['type'].map({'spam':1,'ham':0})
df = df.drop('type',axis=1)

df_train = df.sample(frac=0.8,random_state=42)
df_val = df.drop(index=df_train.index)

cv = CountVectorizer(max_features=1000)
message_train = cv.fit_transform(df_train['message'])
message_val = cv.transform(df_val['message'])

X_train = torch.tensor(message_train.todense(),dtype=torch.float32)
y_train = torch.tensor(df_train['spam'], dtype=torch.float32).reshape(-1,1)

X_val = torch.tensor(message_val.todense(),dtype=torch.float32)
y_val = torch.tensor(df_val['spam'], dtype=torch.float32)



model = nn.Linear(1000,1)
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)


for i in range(0, 10000):

    # Training Pass
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs,y_train)
    loss.backward()
    optimizer.step()

    if i % 100 ==0:
        print(loss)


def evaluate_model(X,y):

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


print("Evaluating the training data.")

evaluate_model(X_train,y_train)

print("Evaluating the validation data")

evaluate_model((X_val,y_val))


