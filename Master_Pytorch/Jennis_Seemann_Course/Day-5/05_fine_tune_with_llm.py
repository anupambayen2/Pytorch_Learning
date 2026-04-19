from transformers import BartTokenizer, BartModel
import torch 
from torch import nn
from tqdm import tqdm

messages =[
    'we have reales a new product, do you want to buy it?',
    'Winner! Great Deal, call us to the product offer free',
    'Tomorrow is my birthday, do you come to the party?',
]

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

X = convertin_to_embeddings(messages)
print(X.shape)