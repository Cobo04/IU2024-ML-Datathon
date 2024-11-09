import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from gensim.models import Word2Vec
import re
import os

# Load dataset
df = pd.read_csv(os.path.join('.', 'data', 'reviews.csv'), encoding='ISO-8859-1')

# Preprocess text data
df['Text'] = df['Text'].str.replace("[^a-zA-Z#]", " ").apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))

# Combine all negative tweets for analysis
neg_words_text = " ".join([sentence for sentence in df['Text'][df['Sentiment'] == 1]])
negative_words = set(neg_words_text.split())

# Train Word2Vec on the entire text with expanded negative words vocabulary
all_text = [sentence.split() for sentence in df['Text']]
w2v_model = Word2Vec(sentences=all_text, vector_size=100, window=5, min_count=2, workers=4)
w2v_model.build_vocab([list(negative_words)], update=True)
w2v_model.train(all_text, total_examples=w2v_model.corpus_count, epochs=10)

# Create average word embeddings for each tweet
def tweet_to_vec(tweet, model):
    words = tweet.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)  # If no words are in the vocabulary
    return np.mean(word_vecs, axis=0)

# Convert each tweet to an embedding vector
embedding_vectors = np.array([tweet_to_vec(tweet, w2v_model) for tweet in df['Text']])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(embedding_vectors, df['Sentiment'], random_state=42, test_size=0.25)

# Train MLPClassifier on embedding vectors
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=300, alpha=1e-4,
                        solver='lbfgs', verbose=True, random_state=1,
                        learning_rate_init=.05, learning_rate='adaptive')
mlp_model.fit(x_train, y_train)

# Test the model
pred = mlp_model.predict(x_test)
print("DONE!")
print("F1 Score:", f1_score(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))
