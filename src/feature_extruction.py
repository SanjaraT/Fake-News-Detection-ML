import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def add_text_features(df):
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x : len(x.split()))
    df['exclamation_count'] = df['text'].apply(lambda x : x.count('!'))
    return df
