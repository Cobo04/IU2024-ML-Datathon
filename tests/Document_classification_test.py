import pandas
import os
from sklearn import model_selection
import csv
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import metrics

'''
reads reviews.csv and stores its contents in experiment_data
experiment_data: first item is text, second item is int
'''
experiment_data = []
with open(os.path.join('.', 'data', 'reviews.csv'), newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',', quotechar='"')
    header = next(datareader)
    for row in datareader:
        if len(row) == 2:
            experiment_data.append( [row[0].strip(), int(row[1].strip())] )

# text and score
data = pandas.read_csv(os.path.join('.', 'data', 'reviews.csv'), encoding='latin-1', header=None, names=["Tweet", (0, 1)])
# just text
dataList = data.iloc[1:, 0].tolist()

'''
makes a list of pos/neg occurrences
'''
data_labels = []
for e in experiment_data:
    if e[1] == 1:
        data_labels.append('pos')
    else:
        data_labels.append('neg')

# print(data.head())

train_text, test_text, train_label, test_label = model_selection.train_test_split(dataList,
                                                                                  data_labels,
                                                                                  test_size=0.33)

encoder = preprocessing.LabelEncoder()

train_label = encoder.fit_transform(train_label)
test_label = encoder.fit_transform(test_label)

vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
vectorizer.fit(dataList)

train_text_count = vectorizer.transform(train_text)
test_text_count = vectorizer.transform(test_text)

classifier = linear_model.LogisticRegression(solver='saga')
classifier.fit(train_text_count, train_label)
predictions = classifier.predict(test_text_count)

accuracy = metrics.accuracy_score(predictions, test_label)
print("LR, Count Vectors: ", accuracy)