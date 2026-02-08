import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def add_text_features(df):
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x : len(x.split()))
    df['exclamation_count'] = df['text'].apply(lambda x : x.count('!'))
    return df

def tfidif_features(train_text, test_text):
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words='english'
    )
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)

    return X_train, X_test, tfidf