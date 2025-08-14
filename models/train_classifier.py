"""
Classifier Trainer

Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Script Execution:
> python train_classifier.py disaster.db classifier.pkl

Arguments Description:
    1) Path to SQLite database
    2) Path to pickle file name
"""
# import libraries
import sys
import os
import re
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

# import nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite database
    Output:
        X -> a dataframe containing features
        y -> a dataframe containing labels
        category_names -> List of categories name
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    print(f'load table {table_name}')
    df = pd.read_sql_table(table_name,engine)

    #Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)
    
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = y.columns # This will be used for visualization purpose
    return X, y, category_names

def tokenize(text):
    """
    Tokenizes the input text by performing the following steps:
    1. Replaces URLs found in the text with a 'urlplaceholder' string.
    2. Tokenizes the text into words using NLTK's word_tokenize.
    3. Lemmatizes each token using WordNetLemmatizer (defaulting to noun if no POS tag is provided).
    4. Converts tokens to lowercase and removes leading/trailing whitespace.

    Args:
        text (str): The input string to be tokenized.

    Returns:
        list: A list of cleaned, lemmatized tokens.
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds a machine learning pipeline for multi-output classification.

    The pipeline consists of three main steps:
    1.  **CountVectorizer:** Converts text data into a matrix of token counts.
        It uses a custom `tokenize` function (assumed to be defined elsewhere)
        to preprocess the text.
    2.  **TfidfTransformer:** Transforms the count matrix into a normalized
        TF-IDF (Term Frequency-Inverse Document Frequency) representation,
        which downweights common words and highlights important ones.
    3.  **MultiOutputClassifier:** Applies an AdaBoostClassifier to each
        output target independently, enabling the model to handle
        multi-label classification tasks.

    Returns:
        sklearn.pipeline.Pipeline: A scikit-learn pipeline object ready for
                                   training and prediction on text data.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
        # specify parameters for grid search - only limited paramter, as the training takes to much time,
    # more testing was done in the jupyter notebooks
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=param_grid,cv=5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        pipeline -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    y_prediction_test = model.predict(X_test)
    
    overall_accuracy = (y_prediction_test == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    
    # Print the whole classification report.
    y_prediction_test = pd.DataFrame(y_prediction_test, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],y_prediction_test[column])) 

def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()