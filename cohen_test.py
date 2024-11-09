import pandas as pd
import os
import contractions
from bs4 import BeautifulSoup
import re
import string
from unidecode import unidecode
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet

df = pd.read_csv(os.path.join('.', 'data', 'reviews.csv'), encoding='ISO-8859-1')
# [Text] [Sentiment]
text_col = df["Text"]

text_col = text_col.apply(lambda x: x.lower())

text_col = text_col.apply(
    lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))

text_col = text_col.apply(
    lambda x: BeautifulSoup(x, 'html.parser').get_text())

text_col = text_col.apply(lambda x: re.sub(r'\d+', '', x))
text_col = text_col.apply(lambda x: re.sub("[.]", " ", x))

print("nshjd")

# '!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~' 32 punctuations in python string module
text_col = text_col.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
text_col = text_col.apply(lambda x: re.sub(' +', ' ', x))

text_col = text_col.apply(lambda x: unidecode(x, errors="preserve"))

text_col = text_col.apply(lambda x: str(TextBlob(x).correct()))

print("continuing")

# nltk.download("stopwords")
sw_nltk = nltk.corpus.stopwords.words('english')
# stopwords customaization: Add custom stopwords
new_stopwords = ['cowboy']
sw_nltk.extend(new_stopwords)
# stopwords customaization: Remove already existing stopwords
sw_nltk.remove('not')
text_col = text_col.apply(
    lambda x: " ".join([ word for word in x.split() if word not in sw_nltk]) )

def download_if_non_existent(res_path, res_name):
    try:
        nltk.data.find(res_path)
    except LookupError:
        print(f'resource {res_name} not found in {res_path}')
        print('Downloading now ...')
        nltk.download(res_name)
download_if_non_existent('corpora/stopwords', 'stopwords')
download_if_non_existent('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
download_if_non_existent('corpora/wordnet', 'wordnet')

nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
text_col = text_col.apply(lambda x: lemmatizer.lemmatize(x))

lemmatizer.lemmatize("leaves") # outputs 'leaf'

lemmatizer.lemmatize("leaves", wordnet.VERB) # outputs 'leave'

nltk.pos_tag(["leaves"]) # outputs [('leaves', 'NNS')]

sentence = "He leaves for England"
pos_list_of_tuples = nltk.pos_tag(nltk.word_tokenize(sentence))
pos_list_of_tuples

pos_tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

sentence = "He leaves for England"
pos_list_of_tuples = nltk.pos_tag(nltk.word_tokenize(sentence))
new_sentence_words = []
for word_idx, word in enumerate(nltk.word_tokenize(sentence)):
    nltk_word_pos = pos_list_of_tuples[word_idx][1]
    wordnet_word_pos = pos_tag_dict.get(nltk_word_pos[0].upper(), None)
    if wordnet_word_pos is not None:
        new_word = lemmatizer.lemmatize(word, wordnet_word_pos)
    else:
        new_word = lemmatizer.lemmatize(word)

    new_sentence_words.append(new_word)
new_sentence = " ".join(new_sentence_words)
print(new_sentence)