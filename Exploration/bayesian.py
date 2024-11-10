# ==============================================
# ===== IMPORTANT INSTALLATION INFORMATION =====
# ==============================================
'''
 1.) For nltk to integrate correctly with Spacy, you MUST install numpy version 1.25.1
     --> !pip install --force-reinstall -v "numpy==1.25.1"
'''

# ==================================
# ===== Necessary Dependencies =====
# ==================================
import os
import csv
import math
import random
import zipfile
import ast
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Make sure NLTK data is downloaded for tokenization and stop words
nltk.download('punkt')
nltk.download('stopwords')

# Define sigmoid function if not in 'secret.py'
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ==============================================
# ===== Load Lexicon Data (VADER Sentiment) ====
# ==============================================

username = "cohen"
nltk_data_folder = f"C:/Users/{username}/AppData/Roaming/nltk_data"
vader_filename = "vader_lexicon/vader_lexicon.txt"
vader_data = {}
with zipfile.ZipFile(os.path.join(nltk_data_folder, "sentiment", 'vader_lexicon.zip')) as z:
    if vader_filename in z.namelist():
        with z.open(vader_filename) as f:
            for l in f:
                tokens = l.decode(encoding='utf-8').strip().split('\t')
                if len(tokens) != 4:
                    continue
                vader_data[tokens[0]] = (float(tokens[1]), float(tokens[2]), ast.literal_eval(tokens[3]))

# ===========================================
# ===== Function Definitions for Features ===
# ===========================================

def generate_feature_vector(text: str) -> np.array:
    tokens = word_tokenize(text)
    scores = [vader_data.get(t, [0, 0]) for t in tokens]
    negative_terms = sum(1 for i in scores if i[0] < 0)
    positive_terms = sum(1 for i in scores if i[0] > 0)
    no_in_text = 1 if "no" in tokens else 0
    pronouns = {"I", "you", "me", "your", "mine"}
    count_pronouns = sum(1 for i in tokens if i in pronouns)
    excl_in_text = 1 if "!" in tokens else 0
    return np.array([positive_terms, negative_terms, no_in_text, count_pronouns, excl_in_text, math.log(len(tokens))])

# Define cross entropy loss
def cross_entropy_loss(y_hat, y):
    return -np.log(y_hat)

# Define stochastic gradient descent function
def stochastic_gradient_descent_silent(data):
    w = np.array([0, 0, 0, 0, 0, 0])
    b = 0
    learning_rate = 0.1
    for text, y in data:
        x = generate_feature_vector(text)
        y_hat = sigmoid(np.dot(w, x) + b)
        gradient_b = y_hat - y
        b = b - learning_rate * gradient_b
        gradient_w = (y_hat - y) * x
        w = w - learning_rate * gradient_w
    return w, b

# ==============================================
# ===== Load and Prepare Dataset for Models ====
# ==============================================

data = pd.read_csv(os.path.join('.', 'data', 'reviews.csv'), encoding='latin-1', header=None, names=["Tweet", "Label"])
dataList = data['Tweet'].tolist()
data_labels = [1 if label == 1 else 0 for label in data["Label"]]

# Vectorize the text data
vectorizer = CountVectorizer(analyzer='word', lowercase=True, stop_words='english', ngram_range=(1, 3))
features = vectorizer.fit_transform(dataList)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    data_labels, 
    train_size=0.80, 
    random_state=1234)

# ===========================================
# ===== Train and Evaluate Models ===========
# ===========================================

# Train Naive Bayes classifier with MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

# Evaluate Naive Bayes Model
print(accuracy_score(y_test, y_pred))

print("Training set size:", len(y_train))
print("Test set size:", len(y_test))
