# ==============================================
# ===== IMPORTANT INSTALLATION INFORMATION =====
# ==============================================
"""
TODO
1. check to see if we can integrate weights and bias
2. figure out how we merge the nltk process to our logistic regression
3. mlp classifier (neural network) 
4. word embeddings to neural network

"""

# ==================================
# ===== Necessary Dependencies =====
# ==================================
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
import pandas
from secret import sigmoid
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import zipfile
import ast
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from joblib import parallel_backend

# If you haven't downloaded these
# import nltk
# nltk.download('vader_lexicon')
# nltk.download('punkt_tab')

# ================================================================
# ===== General Formatting -- Data Settup and Initialization =====
# ================================================================

username = os.getlogin()

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

# =============================
# ===== Package Functions =====
# =============================

def generate_feature_vector(text: str) -> list:
    """
    Finds the positive and negative semantic values of a given text \n\n
    Param [text]: String of the Tweet Text \n\n
    Output [array]: Positive and Negative Semantic Values of Given Text 
    """
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

def generate_pos_neg_sigmoid(text, w=[2.5, -5.0, -1.2, 0.5, 2.0, 0.7], b=0.1):
    """
    Generates the sigmoid value of a text using the weights and bias of the text \n\n
    Param [text]: String of a tweet's text \n\n
    Param [list]: List of weights \n\n
    Param [int]: Bias value \n\n
    Output [tuple]: (positive sigmoid value, negative sigmoid value)
    """
    weights = np.array(w)
    vector = generate_feature_vector(text)

    sigmoid_positive = sigmoid( np.dot(weights, vector) + b )
    sigmoid_negative = 1 - sigmoid_positive

    return (sigmoid_positive, sigmoid_negative)

def cross_entropy_loss(y_hat):
  """
    function \n\n
    Param [int]: y_hat is the sigmoid of the dot-product of the weight and feature vector after adding the bias value \n\n
    Output [type]:
    """
  return -np.log(y_hat)

def generate_pos_neg(text, w=[2.5, -5.0, -1.2, 0.5, 2.0, 0.7], b=0.1):
    """
    function \n\n
    Param [type]:  \n\n
    Output [type]:
    """
    cel_positive = cross_entropy_loss(generate_pos_neg_sigmoid(text, w, b)[0], 1)
    cel_negative = cross_entropy_loss(generate_pos_neg_sigmoid(text, w, b)[1], 1)
    return (cel_positive, cel_negative)


def stochastic_gradient_descent(data, silent=True):
    """
    Generate weights and bias \n\n
    Param [list]: Experiment data \n\n
    Param [boolean]: Print statements through generation process (Default --> True) \n\n
    Output [tuple]: Weights and bias for given dataset
    """
    w = np.array([0, 0, 0, 0, 0, 0])
    b = 0
    learning_rate = 0.1
    for text, y in data:
        x = generate_feature_vector(text)
        if not silent: print("x:", x)
        y_hat = sigmoid( np.dot(w, x) + b )
        if not silent: print("y_hat:", y_hat)
        gradient_b = y_hat - y
        if not silent: print("gradient b:", gradient_b)
        b = b - learning_rate * gradient_b
        if not silent: print("new gradient b:", b)
        gradient_w = (y_hat - y) * x
        if not silent: print("gradient w:", gradient_w)
        # w = gradient_w - learning_rate * gradient_w
        w = w - learning_rate * gradient_w
        if not silent: print("new gradient w:", w)
    return w, b

def generate_experiment_data():
    """
    function \n\n
    Param [type]:  \n\n
    Output [type]:
    """
    experiment_data = []

    with open(os.path.join('.', 'data', 'reviews.csv'), newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='"')
        header = next(datareader)
        for row in datareader:
            if len(row) == 2:
                experiment_data.append( [row[0].strip(), int(row[1].strip())] )

    return experiment_data

def output_pos_neg_exp_data_sum(experiment_data):
    """
    function \n\n
    Param [type]:  \n\n
    Output [type]:
    """
    count_positive = sum([ 1 for x in experiment_data if x[1] == 1 ])
    count_negative = sum([ 1 for x in experiment_data if x[1] == 0 ])
    print(f"Positive: {count_positive}\t Negative: {count_negative}")
    print("Total reviews:", len(experiment_data))

def generate_data_and_labels():
    data = pandas.read_csv(os.path.join('.', 'data', 'reviews.csv'), encoding='latin-1', header=None, names=["Tweet", (0, 1)])
    dataList = data.iloc[1:, 0].tolist()
    return data, dataList

def generate_data_labels(experiment_data):
    data_labels = []
    for e in experiment_data:
        if e[1] == 1:
            data_labels.append('pos')
        else:
            data_labels.append('neg')
    return data_labels

def execute_model(dataList, data_labels):
    vectorizer = CountVectorizer(analyzer = 'word', lowercase = True, stop_words='english', ngram_range=(1, 3))
    features = vectorizer.fit_transform(dataList)

    X_train, X_test, y_train, y_test  = train_test_split(
            features, 
            data_labels,
            train_size=0.80, 
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
    df = pandas.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['Antisimetic Actual','Not Antisimetic Actual'], columns=['Antisimetic Predicted','Not Antisimetic Predicted'])

    print(df)

    return accuracy_score(y_test, y_pred)

def execute_model_RF_with_dimensionality_reduction(dataList, data_labels):
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(
            analyzer='word',
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)  # Reduced ngram range for less dimensionality and faster training
        )),
        ('svd', TruncatedSVD(n_components=100)),
        ('classifier', RandomForestClassifier(random_state=42))
    ], verbose=True)

    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [2]
    }

    X_train, X_test, y_train, y_test = train_test_split(
        dataList,
        data_labels,
        train_size=0.80,
        random_state=1234
    )

    # Use thread-based parallelism
    with parallel_backend('threading', n_jobs=1):
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_

    # Cross-validation without parallelism
    cv_scores = cross_val_score(
        best_pipeline, dataList, data_labels, cv=5, n_jobs=1, verbose=2
    )

    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", cv_scores.mean())

    y_pred = best_pipeline.predict(X_test)

    print("Start prediction with Random Forest model (with dimensionality reduction)")

    # Prediction matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    df = pandas.DataFrame(
        cm,
        index=['Antisemetic Actual', 'Not Antisemetic Actual'],
        columns=['Antisemetic Predicted', 'Not Antisemetic Predicted']
    )

    print(df)

    return accuracy_score(y_test, y_pred)

def execute_model_MLP(dataList, data_labels):
    X = dataList
    y = data_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cv = CountVectorizer()

    X_train_cv = cv.fit_transform(X_train)

    X_test_cv = cv.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=300, alpha=1e-4,
                        solver='lbfgs', verbose=True, random_state=1,
                        learning_rate_init=.05, learning_rate='adaptive')

    # params = {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 'auto', 'beta_1': 0.9, 
    #           'beta_2': 0.999, 'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': (100,), 
    #           'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 200, 'momentum': 0.9, 
    #           'n_iter_no_change': 10, 'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': None, 'shuffle': True, 'solver': 'adam', 
    #           'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}

    # mlp = MLPClassifier()
    # mlp.set_params(**params)
    
    print("Training...")
    mlp.fit(X_train_cv, y_train)

    predictions = mlp.predict(X_test_cv)

    print(f"Training set score: {mlp.score(X_train_cv, y_train):.5f}")
    print(f"Test set score: {mlp.score(X_test_cv, y_test):.5f}")

    print(metrics.confusion_matrix(y_test, predictions))
    df = pandas.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['Antisimetic Actual','Not Antisimetic Actual'], columns=['Antisimetic Predicted','Not Antisimetic Predicted'])
    print(df)

def fine_tune_model_parameters(dataList, data_labels):

    print("Vectorizing data")
    vectorizer = CountVectorizer(analyzer = 'word', lowercase = True, stop_words='english', ngram_range=(1, 3), max_features=50000)
    features = vectorizer.fit_transform(dataList)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        data_labels,
        train_size=0.20,  # Smaller subset for benchmarking
        random_state=1234
    )
    print("Data split complete. Training samples:", X_train.shape[0], "Testing samples:", X_test.shape[0])

    print("Initializing GridSearchCV with parameter grid...")
    from sklearn.model_selection import GridSearchCV
    params = {
        'C': [3, 5, 7, 10],      # Fine-tuning C (best val was 5)
        'max_iter': [100, 200],  # Allow more iterations
        'class_weight': [None, 'balanced']
    }
    params = {}
    # grid_search = GridSearchCV(LogisticRegression(warm_start=True, tol=1e-4, solver='lbfgs'), param_grid=params, cv=3, verbose=1)
    grid_search = GridSearchCV(MLPClassifier(), param_grid=params, cv=3, verbose=1)
    print("Starting grid search...")

    grid_search.fit(X_train, y_train)
    print("Grid search complete.")
    print("Best parameters found:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    print("Best model initialized with parameters:", best_model.get_params())

    print("Starting prediction on the test set...")
    y_pred = best_model.predict(X_test)

    print("Evaluation metrics:")
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

    df = pandas.DataFrame(
        metrics.confusion_matrix(y_test, y_pred), 
        index=['Antisemetic Actual', 'Not Antisemetic Actual'], 
        columns=['Antisemetic Predicted', 'Not Antisemetic Predicted']
    )
    print(df)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of the model:", accuracy)

    return accuracy



# ========================================
# ===== Automatically Generate Model =====
# ========================================

def auto_generate_model():
    return execute_model(generate_data_and_labels()[1], generate_data_labels(generate_experiment_data()))

# ================
# ===== Main =====
# ================

print(auto_generate_model())
# execute_model_MLP(generate_data_and_labels()[1], generate_data_labels(generate_experiment_data()))

# fine_tune_model_parameters(generate_data_and_labels()[1], generate_data_labels(generate_experiment_data()))
# execute_model_RF_with_dimensionality_reduction(generate_data_and_labels()[1], generate_data_labels(generate_experiment_data()))