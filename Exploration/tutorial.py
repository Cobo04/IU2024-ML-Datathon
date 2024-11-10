import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import warnings
import os
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

warnings.filterwarnings('ignore')

# ==================================
# ======= Loading Dataset ==========
# ==================================
df = pd.read_csv(os.path.join('.', 'data', 'reviews.csv'), encoding='ISO-8859-1')

# ==================================
# ======== Preprocessing ===========
# ==================================

# removes pattern in the input text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

# remove special chars, numbers, and punctuation
df['Text'] = df['Text'].str.replace("[^a-zA-Z#]", " ")

# remove short words
df['Text'] = df['Text'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))

# individual words considered as tokens
tokenized_tweet = df['Text'].apply(lambda x: x.split())

# stem the words
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])

# combine words into single sentence
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
df['Text'] = tokenized_tweet

# ==================================
# ========= EDA - Negative Words ============
# ==================================

# Join all negative tweets into one text for analysis
neg_words_text = " ".join([sentence for sentence in df['Text'][df['Sentiment'] == 1]])

# Identify the most common negative words
neg_word_counts = Counter(neg_words_text.split())
common_neg_words = [word for word, count in neg_word_counts.most_common(20)]  # Top 20 frequent negative words

# Add binary features for common negative words
for word in common_neg_words:
    df[f'word_{word}'] = df['Text'].apply(lambda x: 1 if word in x.split() else 0)

# ==================================
# ========= Feature Extraction with Custom Dictionary ============
# ==================================

# Include custom dictionary words in the CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# Fit and transform text data
bow = bow_vectorizer.fit_transform(df['Text'])

# Add new binary features to the Bag-of-Words features
additional_features = df[[f'word_{word}' for word in common_neg_words]].values
bow = np.hstack((bow.toarray(), additional_features))

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(bow, df['Sentiment'], random_state=42, test_size=0.25)

# ==================================
# ======== Model Training ==========
# ==================================

# Train logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Testing
pred = model.predict(x_test)
print("DONE!")
print("F1 Score:", f1_score(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))
