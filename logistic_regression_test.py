"""
TODO
- make function to generate sigmoid weights bias with a given vector 
- generate sigmoid positive, sigmoid negative, and cell losses 
- create a list of the first columnn of a pandas dataframe
- modify weights and find a different algorithm? compare that to stochastic gradient descent
- bayesian classifier?
- find a different cassifier/regressor model?

"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score
from collections import Counter
import os
import csv
import math
import random
import pandas
from secret import sigmoid
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import zipfile
import ast

# import nltk
# nltk.download()

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

def generate_feature_vector(text: str) -> list:
    tokens = word_tokenize(text)
    scores = [ vader_data.get(t, [0, 0]) for t in tokens ]
    negative_terms = sum(1 for i in scores if i[0] < 0)
    positive_terms = sum(1 for i in scores if i[0] > 0)
    if "no" in tokens:
        no_in_text = 1
    else:
        no_in_text = 0
    pronouns = set( ("I", "you", "me", "your", "mine") )
    count_pronouns = sum(1 for i in tokens if i in pronouns)
    if "!" in tokens:
        excl_in_text = 1
    else:
        excl_in_text = 0
    return np.array([positive_terms, negative_terms, no_in_text, count_pronouns, excl_in_text, math.log(len(tokens))])

# vector = generate_feature_vector("I slightly dislike my mom, but its fine")
# #temp weights 
# weights = np.array([2.5, -5.0, -1.2, 0.5, 2.0, 0.7])
# b = 0.1

# sigmoid_positive = sigmoid( np.dot(weights, vector) + b )
# sigmoid_negative = 1 - sigmoid_positive

# print(sigmoid_positive, sigmoid_negative)

def cross_entropy_loss(y_hat, y):
  return -np.log(y_hat)

# cel_positive = cross_entropy_loss(sigmoid_positive, 1)
# print(f"sigmoid {sigmoid_positive} for y={1} Loss positive: {cel_positive}")

# cel_negative = cross_entropy_loss(sigmoid_negative, 0)
# print(f"sigmoid {sigmoid_negative} for y={0} Loss positive: {cel_negative}")

def stochastic_gradient_descent(data):
    w = np.array([0, 0, 0, 0, 0, 0])
    b = 0
    learning_rate = 0.1
    for text, y in data:
        x = generate_feature_vector(text)
        print("x:", x)
        y_hat = sigmoid( np.dot(w, x) + b )
        print("y_hat:", y_hat)
        gradient_b = y_hat - y
        print("gradient b:", gradient_b)
        b = b - learning_rate * gradient_b
        print("new gradient b:", b)
        gradient_w = (y_hat - y) * x
        print("gradient w:", gradient_w)
        # w = gradient_w - learning_rate * gradient_w
        w = w - learning_rate * gradient_w
        print("new gradient w:", w)
    return w, b

def stochastic_gradient_descent_silent(data):
    w = np.array([0, 0, 0, 0, 0, 0])
    b = 0
    learning_rate = 0.1
    for text, y in data:
        x = generate_feature_vector(text)
        y_hat = sigmoid( np.dot(w, x) + b )
        gradient_b = y_hat - y
        b = b - learning_rate * gradient_b
        gradient_w = (y_hat - y) * x
        w = w - learning_rate * gradient_w
    return w, b

# data = [("i think chocolate is delicious", 1)]
# w, b = stochastic_gradient_descent_silent(data)

# Automate the process of training the weights

experiment_data = []

with open(os.path.join('.', 'data', 'reviews.csv'), newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',', quotechar='"')
    header = next(datareader)
    for row in datareader:
        if len(row) == 2:
            experiment_data.append( [row[0].strip(), int(row[1].strip())] )

count_positive = sum([ 1 for x in experiment_data if x[1] == 1 ])
count_negative = sum([ 1 for x in experiment_data if x[1] == 0 ])
print(f"Positive: {count_positive}\t Negative: {count_negative}")
print("Total reviews:", len(experiment_data))

# w, b = [ 1.65087395  0.49448245 -0.0074589   0.84723613 -0.00302559  0.96924161], 0.2061591334513687
w, b = stochastic_gradient_descent_silent(experiment_data[:50])
print(w, b)


def test(data, w, b):
    res = []
    for text, y in data:
        x = generate_feature_vector(text)
        y_hat = sigmoid( np.dot(w, x) + b )
        if y_hat > .5:
            y_hat = 1
        else:
            y_hat = 0
        res.append( (y, y_hat) )
    return res

result = test(experiment_data[-20:], w, b)
counts = Counter(result)

print(counts)

# Generate the prediction matrix


data = pandas.read_csv(os.path.join('.', 'data', 'reviews.csv'), encoding='latin-1', header=None, names=["Tweet", (0, 1)])
dataList = data.iloc[1:, 0].tolist()
# dataList = list(data['Tweet'])

# data_labels = data.iloc[:, 1].tolist()
# data.head()
# print(data)

# data = []
data_labels = []
for e in experiment_data:
    if e[1] == 1:
        data_labels.append('pos')
    else:
        data_labels.append('neg')

# print(data)
# print(data_labels)
 
vectorizer = CountVectorizer(analyzer = 'word', lowercase = True, stop_words='english', ngram_range=(1, 3))
features = vectorizer.fit_transform(dataList)

X_train, X_test, y_train, y_test  = train_test_split(
        features, 
        data_labels,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression(max_iter=3000)

log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)

print(accuracy_score(y_test, y_pred))


# Linear Support Vector Classifier (SVC)
# lin_svc_model = LinearSVC(max_iter=3000)
# lin_svc_model = lin_svc_model.fit(X=X_train, y=y_train)
# y_pred_lin_svc = lin_svc_model.predict(X_test)
# print(accuracy_score(y_test, y_pred_lin_svc))