import numpy as np
import pandas as pd

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('datasets/sentence1.csv')

from bs4 import BeautifulSoup
import re
import contractions

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text
df['sentences_processed'] = df['sentences'].apply(lambda sentence: strip_html_tags(sentence))

def remove_links(text):    
    remove_https = re.sub(r'http\S+', '', text)
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    return remove_com
df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: remove_links(sentence))

def lower_casing_text(text):
    text = text.lower()
    return text
df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: lower_casing_text(sentence))

def expand_contractions(text):
    return ' '.join([contractions.fix(word) for word in text.split()])
df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: expand_contractions(sentence))

def removing_special_characters(text):
    Formatted_Text = re.sub(r"[^a-zA-Z0-9]+", ' ', text) 
    return Formatted_Text
df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: removing_special_characters(sentence))

def remove_whitespace(text):
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', text)
    return Without_whitespace
df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: remove_whitespace(sentence))

df = df[df['sentences_processed'].str.len() > 0].reset_index(drop=True)

df_clean = df[['label', 'sentences_processed']]
DF_train_valid, DF_test = np.split(df_clean.sample(frac=1, random_state=42), [int(.8*len(df_clean))])
DF_train, DF_valid, DF_test = np.split(df_clean.sample(frac=1, random_state=42), [int(.6*len(df_clean)), int(.8*len(df_clean))])
X_train = DF_train['sentences_processed']
Y_train = DF_train['label']
X_valid = DF_valid['sentences_processed']
Y_valid = DF_valid['label']
X_test = DF_test['sentences_processed']
Y_test = DF_test['label']

from transformers import BertTokenizer
from transformers import BertModel

from torch import nn
from torch.optim import Adam
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [label for label in df['label']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['sentences_processed']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

max_recall = 0

def train(model, learning_rate, epochs):
    train_data, val_data = np.split(DF_train_valid.sample(frac=1), [int(.75*len(DF_train_valid))])

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_tp_train = 0
        total_fp_train = 0
        total_pos_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())

            tp = ((output.argmax(dim=1) == 1) & (train_label == 1)).sum().item()
            fp = ((output.argmax(dim=1) == 1) & (train_label == 0)).sum().item()
            pos = (train_label == 1).sum().item()
            total_tp_train += tp
            total_fp_train += fp
            total_pos_train += pos

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_tp_val = 0
        total_fp_val = 0
        total_pos_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                
                tp = ((output.argmax(dim=1) == 1) & (val_label == 1)).sum().item()
                fp = ((output.argmax(dim=1) == 1) & (val_label == 0)).sum().item()
                pos = (val_label == 1).sum().item()
                total_tp_val += tp
                total_fp_val += fp
                total_pos_val += pos
        
        torch.save(model, '../models/transformer_' + str(epoch_num) + '.pt')

        print(
            f'Epochs: {epoch_num + 1} \
            | Train Precision: {total_tp_train / (total_tp_train + total_fp_train): .3f} \
            | Train Recall: {total_tp_train / total_pos_train: .3f} \
            | Val Precision: {total_tp_val / (total_tp_val + total_fp_val): .3f} \
            | Val Recall: {total_tp_val / total_pos_val: .3f}')
                  
EPOCHS = 10
model = BertClassifier()
LR = 1e-6
              
train(model, LR, EPOCHS)