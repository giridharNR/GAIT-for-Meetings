import sys

option = sys.argv[1]
method = ''

if option == '-ann':
    method = 'ann'
elif option == '-rf':
    method = 'rf'
elif option == '-tr':
    method = 'tr'
else:
    print('Pass -ann / -rf / -tr as the first argument to select a model.')

filename = sys.argv[2]
outfile = sys.argv[3]

import pandas as pd
import torch

from transformers import BertTokenizer
from transformers import BertModel

from torch import nn
from tqdm import tqdm

from bs4 import BeautifulSoup
import re
import contractions

from torch.utils.data import DataLoader

from joblib import dump, load

from transformers import pipeline

def loadModel(modelName):
    if modelName == 'ann':
        return torch.load('bow_ann.pt')
    elif modelName == 'rf':
        return load('random_forest.joblib')
    elif modelName == 'tr':
        return torch.load('transformer.pt')

model = loadModel(method)

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_links(text):    
    remove_https = re.sub(r'http\S+', '', text)
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    return remove_com

def lower_casing_text(text):
    text = text.lower()
    return text

def expand_contractions(text):
    return ' '.join([contractions.fix(word) for word in text.split()])  

def removing_special_characters(text):
    Formatted_Text = re.sub(r"[^a-zA-Z0-9]+", ' ', text) 
    return Formatted_Text

def remove_whitespace(text):
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', text)
    return Without_whitespace

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['sentences_processed']]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)

        return batch_texts, idx

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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda and method != 'rf':
    model = model.cuda()

f = open(filename)
paragraphs = f.readlines()
f.close()
out = open(outfile, 'w')

for paragraph in paragraphs:
    paragraph = paragraph[:-1]
    sentences = re.split(r'\.|\(PERSON.*?\)',paragraph)
    df = pd.DataFrame()
    for idx, sentence in enumerate(sentences):
        df.at[idx,'sentences'] = sentence

    df['sentences_processed'] = df['sentences'].apply(lambda sentence: strip_html_tags(sentence))
    df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: remove_links(sentence))
    df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: lower_casing_text(sentence))
    df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: expand_contractions(sentence))
    df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: removing_special_characters(sentence))
    df['sentences_processed'] = df['sentences_processed'].apply(lambda sentence: remove_whitespace(sentence))
    df = df[df['sentences_processed'].str.len() > 0].reset_index(drop=True)

    if method == 'tr':
        dataloader = torch.utils.data.DataLoader(Dataset(df), batch_size=1)

        with torch.no_grad():
            for val_input, idx in dataloader:
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                df.at[idx.item(),'classified_label'] = output.argmax(dim=1)[0].item()
    elif method == 'ann':
        print()
        #ann
    elif method == 'rf':
        X = df['sentences_processed']
        pred = model.predict(X)
        df['classified_label'] = pred

    df_action = df[df['classified_label'] == 1].reset_index(drop=True)

    actions = ""
    for sentence in df_action['sentences']:
        actions = actions + sentence.strip().capitalize() + '. '

    out.write(actions + '\n')

out.close()