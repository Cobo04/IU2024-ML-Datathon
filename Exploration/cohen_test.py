from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.naive_bayes import GaussianNB
import os
import csv
import math
import random
import pandas as pd
import re
# from secret import sigmoid
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import nltk
import zipfile
import ast
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from joblib import parallel_backend
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

data = pd.read_csv(os.path.join('.', 'data', 'reviews.csv'), encoding='ISO-8859-1')
data = data.dropna(subset=['Text', 'Sentiment']) # Remove/drop NaN values
texts = data['Text'].tolist()
labels = data['Sentiment'].tolist()

def clean_tweet(textList):
    """Takes in raw text in form of list from pandas database"""
    for i in range(len(textList)):
        text = textList[i]
        text = re.sub(r'^RT[\s]+', '', text)         # Remove "Old Style" Retweet
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text) #Remove Hyperlinks
        text = re.sub(r'http\S+', '', text)          # Remove URLs
        text = re.sub(r'@\w+', '', text)             # Remove mentions
        text = re.sub(r'#\w+', '', text)             # Remove hashtags
        text = re.sub(r'\s+', ' ', text).strip()     # Remove excess whitespace
        text = text.lower()
        textList[i] = text
        # print(text)
    return textList

def tokenize_tweet(textList):
    """Takes in a cleaned set of tweets utilizing the clean_tweet function"""
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    stopwords_english = stopwords.words('english') 
    stemmer = PorterStemmer()

    # Tokenize the tweets
    for i in range(len(textList)):
        text = textList[i]
        text_token = tokenizer.tokenize(text)
        tweets_clean = []
        for word in text_token:
            if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
                tweets_clean.append(word)

        # Stem the tweets
        tweets_stem = []
        for word in tweets_clean:
            stem_word = stemmer.stem(word)  # stemming word
            tweets_stem.append(stem_word)  # append to the list
        textList[i] = (" ".join(tweets_stem)).strip()
    return textList

def execute_model(dataList, data_labels):
    vectorizer = CountVectorizer(analyzer = 'word', lowercase = True, stop_words='english', ngram_range=(1, 3))
    features = vectorizer.fit_transform(dataList)

    X_train, X_test, y_train, y_test  = train_test_split(
            features, 
            data_labels,
            test_size=0.20, 
            random_state=1234)

    print("Dataset Splitting Complete")

    # log_model = LogisticRegression(max_iter=500, penalty='elasticnet', C=10, solver='saga', l1_ratio=1)

    # These are the best parameters found from the function below for model initialization (0.8934)
    # params = {'C': 3, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 
    #           'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'deprecated', 
    #           'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 
    #           'verbose': 0, 'warm_start': True}
    
    # print("Initializing Model")
    # log_model = LogisticRegression()

    # print("Loading Parameters")
    # log_model.set_params(**params)

    log_model = LogisticRegression(max_iter=300, C=5, solver='lbfgs', class_weight= None)

    print("Training...")
    log_model = log_model.fit(X=X_train, y=y_train)

    y_pred = log_model.predict(X_test)

    #prediction matrix
    predictions = log_model.predict(X_test)
    print(metrics.confusion_matrix(y_test, y_pred))
    df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['Antisimetic Actual','Not Antisimetic Actual'], columns=['Antisimetic Predicted','Not Antisimetic Predicted'])

    print(df)

    return accuracy_score(y_test, y_pred)

# print(execute_model(texts, labels))
print(tokenize_tweet(clean_tweet(["I love my mom and I think !!that riding my bicycle is actually rREALLDYB !!!!"])))