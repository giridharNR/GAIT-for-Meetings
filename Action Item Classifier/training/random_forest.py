import numpy as np
import pandas as pd

df = pd.read_csv('datasets/sentence1.csv')

from bs4 import BeautifulSoup
import re
import contractions

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

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

rf = model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
rf.fit(X_train,Y_train)
dump(rf, '../models/random_forest.joblib')