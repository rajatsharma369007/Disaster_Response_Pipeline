# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import string
import numpy as np
import joblib
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    This function splits the dataset into independent and target variables and 
    creates a list of category names
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', con=engine)
    
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    categories = Y.columns.tolist()
    
    return X, Y, categories
    

def tokenize(text):
    '''
    This function helps in tokenizing each message
    '''
    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = nltk.corpus.stopwords.words("english")
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return lemmatized


def build_model():
    '''
    This function helps in building the model.
    Creating the pipeline
    Applying Grid search
    Return the model
    '''
    # creating pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # parameters
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.8, 1.0),
    'vect__max_features': (None, 10000),
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__learning_rate': [0.1, 1.0]
    }
    
    # grid search
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function helps in evaluating the trained model using the test set
    '''
    test_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[category_names[i]], test_pred[:, i]))


def save_model(model, model_filepath):
    joblib.dump(model.best_estimator_, model_filepath)


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
