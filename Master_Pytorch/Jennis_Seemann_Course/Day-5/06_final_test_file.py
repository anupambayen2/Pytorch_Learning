from transformers import BartTokenizer, BartModel
import sys
import torch 
from torch import nn
from tqdm import tqdm

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartModel.from_pretrained("facebook/bart-base")

def convertin_to_embeddings(messages):
    embedding_list = []
    for message in tqdm(messages):

        out = tokenizer(
            [message],
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            bart_model.eval()

            pred = bart_model(
                input_ids = out["input_ids"],
                attention_mask = out["attention_mask"]
            )
            embeddings = pred.last_hidden_state.mean(dim=1).reshape(-1)
            embedding_list.append(embeddings)

    return torch.stack(embedding_list)


df = pd.read_csv('./data/SMSSpamCollection',sep='\t',names=['type','message'])

df['spam'] = df['type'].map({'spam':1,'ham':0})
df = df.drop('type',axis=1)

df_train = df.sample(frac=0.8,random_state=42)
df_val = df.drop(index=df_train.index)



X_train = convertin_to_embeddings(df_train['message'].tolist())
X_val = convertin_to_embeddings(df_val['message'].tolist())

# X_train = torch.tensor(message_train.todense(),dtype=torch.float32)
# y_train = torch.tensor(df_train['spam'].values, dtype=torch.float32).reshape(-1,1)

y_train = torch.tensor(df_train['spam'].values,dtype=torch.float32).reshape(-1,1)
y_val = torch.tensor(df_val['spam'].values, dtype=torch.float32).reshape(-1,1)

model = nn.Linear(768,1)
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


print("")
print("-----------eval train data-------------")

evaluate_model(X_train,y_train)

print("")
print("-----------eval validation data--------")

evaluate_model(X_val,y_val)
print("")


X_custom = convertin_to_embeddings([
    'we have reales a new product, do you want to buy it?',
    'Winner! Great Deal, call us to the product offer free',
    'Tomorrow is my birthday, do you come to the party?'
])

model.eval()
with torch.no_grad():
    pred = nn.functional.sigmoid(model(X_custom))
    print(pred)


