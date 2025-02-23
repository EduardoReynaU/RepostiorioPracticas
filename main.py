import torch
import numpy as np                                                                                  
import pandas as pd
#Importing the dataset

with open('MiVecinoTotoScript.txt', 'r') as f:
    data = f.read().split()

#print(data[:10])
#print(len(data))

#Converting the data into embeddings
stoi = {s: i for i, s in enumerate(sorted(set(data)))}
itos = {i: s for s, i in stoi.items()}

data = np.array([stoi[s] for s in data])

data = torch.tensor(data, dtype=torch.long)
#print(data[:10])
#print(data.shape)


#Get embddings
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F

emd_data = nn.Embedding(len(stoi), 30)
emb_data = emd_data(data)
skaler = MinMaxScaler(feature_range=(-1, 1))
emb_data = skaler.fit_transform(emb_data.detach().numpy())
#print(emb_data[:10])
#print(emb_data.shape)

#Padding

pad_rows = 2

emb_data = torch.tensor(emb_data, dtype=torch.float32)
emb_data = F.pad(emb_data, (0, 0, 0,pad_rows))
#print(emb_data.shape)


#Reshape the data

emb_data = emb_data.reshape(8, -1, 30)
#print(emb_data.shape)
#print(emb_data[:2])

X = emb_data
print(X.shape)

#Matriz de pesos de Query
Q_w = torch.randn(30, 30, requires_grad=True)
Q_b = torch.randn(1, 30, requires_grad=True)

#Matriz de pesos de Key

K_w = torch.randn(30, 30, requires_grad=True)
K_b = torch.randn(1, 30, requires_grad=True)

#Matriz de pesos de Value

V_w = torch.randn(30, 30, requires_grad=True)
V_b = torch.randn(1, 30, requires_grad=True)

#Producto de matrices

Q = torch.matmul(X, Q_w) + Q_b
K = torch.matmul(X, K_w) + K_b
V = torch.matmul(X, V_w) + V_b

#print(Q.shape)
#print(K.shape)
#print(V.shape)

K_t = torch.transpose(K, 1, 2)
#print(K_t.shape)

#Multiplicación de Q y K
QK = torch.matmul(Q, K_t)

scores = QK / np.sqrt(X.shape[2])
#print(scores.shape)

#Masking

mask = torch.triu(torch.ones(scores.shape[0], scores.shape[1], scores.shape[2] ), diagonal=1)
mask = mask.bool()

scores.masked_fill_(mask, -float('inf'))
#print(scores)


#Softmax

attn = F.softmax(scores, dim=-1)
print(attn.shape)

#Output

output = torch.matmul(attn, V)
print(output.shape)

#Mulithead Attention

n_heads = 6

