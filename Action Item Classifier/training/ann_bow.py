import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/sentence1.csv')

df['sentences'] = df['sentences'].str.lower()
df['sentences'] = df['sentences'].str.replace('[^a-z]', ' ')
df['Split'] = df['sentences'].apply(lambda x:len(str(x).split()))

X_train, X_valid, Y_train, Y_valid= train_test_split(df['sentences'], df['label'], test_size=0.2, stratify = df['label'].tolist(), random_state=0)

train_dat =list(zip(Y_train,X_train))
valid_dat =list(zip(Y_valid,X_valid))
test_dat=list(zip(df['sentences'].tolist(),df['sentences'].tolist()))

import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
train_iter = train_dat
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) 

def collate_batch(batch):
    print(batch)
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

from torch import nn
import torch.nn.functional as F

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim,64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
train_iter1 = train_dat
num_class = len(set([label for (label, text) in train_iter1]))
print(num_class)
vocab_size = len(vocab)
emsize = 128
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

import time

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        #sys.exit()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count, total_loss = 0, 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            #print(predited_label)
            loss = criterion(predited_label, label)
            total_loss += loss
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count, total_loss/total_count

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
# Hyperparameters
EPOCHS = 10 # epoch
LR =10  # learning rate
BATCH_SIZE = 16 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

train_iter2 = train_dat
test_iter2 =test_dat 
valid_iter2= valid_dat

train_dataloader = DataLoader(train_iter2, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_iter2, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter2, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

T = []
V = []

def performace() :
    model.eval()
    total_acc, total_count, total_loss = 0, 0, 0
    
    _0_1 = 0
    _0_0 = 0
    _1_0 = 0
    _1_1 = 0
    
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(valid_dataloader):
            predited_label = model(text, offsets)
            #print(predited_label)
            
            for I in range(0, label.shape[0]):
                if label[I].item() == 0 :
                    if predited_label.argmax(1)[I].item() == 0 :
                        _0_0 += 1
                    else:
                        _0_1 += 1
                else:
                    if predited_label.argmax(1)[I].item() == 1 :
                        _1_1 += 1
                    else:
                        _1_0 += 1
    
    tp = _0_0
    fn = _0_1
    fp = _1_0
    tn = _1_1
            
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    
    print(prec, rec)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val, A = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
       
    val_acc, A = evaluate(valid_dataloader)
    T.append(accu_val)
    V.append(val_acc)
    
    #print(accu_val, val_acc)

    performace()

torch.save(model, '../models/bow_ann.pt')