# Utility functions
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_data(path):
    df = pd.read_csv(path)
    return df['text'], df['label']

def preprocess(texts):
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def prepare_datasets(path):
    texts, labels = load_data(path)
    X, vectorizer = preprocess(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer