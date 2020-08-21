import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

# sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import recall_score, make_scorer, classification_report, f1_score, precision_recall_fscore_support

# Package for stratified multiclass train-test split 
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .features import LengthExtractor

#nltk
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def load_data(database_filepath, table_name):
    """
    Reads data from table at provided location, splits data into features and labels, and 
    returns respecitve dataframes and additionally a list of label names. 

    Parameters
    ----------
    database_filepath : str
        Path to database including database name.
    table_name : str
        Database table name to select data from.

    Returns
    -------
        X : pandas Series object
            Series containing the text input. 
        Y : pandas DataFrame object
            Dataframe containing the label values.
        categories : list of str
            List of label names in Y.
    """

    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql(table_name, engine)
    X = df.iloc[:, 0]
    Y = df.iloc[:, 3:]
    categories = list(Y.columns)

    return X, Y, categories

def tokenize(text):
    """
    Removes non-alphabetic characters, lemmatizes, and removes stopwords from 
    a text and returns the cleaned tokens in a list. 

    Parameters
    ----------
    text : str
        Text input to be be cleaned.

    Returns
    -------
    tokens_cleaned : list of str
        The cleaned tokens as a list of strings.
    """

    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = word_tokenize(text)

    lemmatizer = WordNetLemmatizer() # Initialize lemmatizer
    tokens_lemmatized = [lemmatizer.lemmatize(token).lower().strip() for token in words]

    tokens_cleaned = [token for token in tokens_lemmatized 
                      if token not in stopwords.words("english")]

    return tokens_cleaned


def stratified_multilabel_train_test_split(X, Y, test_size=0.2):
    """
    Returns train-test-splits of X and Y using MultilabelStratifiedShuffleSplit. This approach 
    tries to ensure that the labels are equally distributed in the train and test set.

    Parameters
    ----------
    X : pandas Series object
        Series containing the text input. 
    Y : pandas DataFrame object
        Dataframe containing the label values.
    test_size : double, default=0.2
        Specifies ratio of test_size.

    Returns
    -------
    X_train : pandas Series object
        Series containing the training messages. 
    X_test : pandas Series object
        Series containing the test messaged. 
    Y_train : pandas DataFrame object
        Dataframe containing the corresponding test labels to X_train.
    Y_test : pandas DataFrame object
        Dataframe containing the corresponding test labels to X_test.
    """

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in msss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    return X_train, X_test, Y_train, Y_test


def make_pipline(classifier):
    """
    Returns pipeline including predefined feature extraction and provided classifier.

    Returns
    -------
    Pipeline object
        Pandas Pipeline object with random forest classifier.

    """
    return Pipeline([
                   ("features", FeatureUnion([
                     ("text_pipeline", Pipeline([
                         ("count_vect", CountVectorizer(tokenizer=tokenize)),
                         ("tfidf", TfidfTransformer())])
                     ),
                     ("length_extractor", LengthExtractor())
                     ])),
                 ("clf", classifier)
                 ])


def grid_search(X_train, y_train, pipeline, param_grid):
    """
    Runs a grid search and returns a grid search object from which the best estimator can be 
    extracted.

    Parameters
    ----------
    X_train : pandas Series object
        Specifies the features of the training dataset.
    Y_train : pandas Dataframe object
        Specifies the lables of the training dataset.
    pipeline : pandas Pipeline object
        Pipeline object to fit with provided hyperparameters.
    param_grid : dict
        Hyperparameters of the pipeline to evaluate

    Returns
    -------
    cv : object
        GridSearchCV object fitted to the training data.

    """
    f1_scorer = make_scorer(f1_score, greater_is_better=True, average='macro')
    msss = MultilabelStratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)

    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, cv=msss, scoring=f1_scorer, n_jobs=-1)
    cv.fit(X_train, y_train)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints scores of provided predition results using sklearn's classifation_report function.
    
    Parameters
    ----------
    Y_pred : pandas Dataframe object, numpy array
        Dataframe with predicted values.
    Y_test : pandas Dataframe object, numpy array
        Dataframe with true values.
    category_name : list of str
        Label names to associate values and categories.

    """
    Y_pred = model.predict(X_test)

    for col_index in range(Y_test.shape[1]):
        print(category_names[col_index])
        scores = classification_report(Y_test.values[:, col_index], Y_pred[:, col_index])
        print(scores)


def save_model(model, model_name):
    """
    Saves the provided model as a pickle file in the models folder.

    Parameters
    ----------
    model : estimator
        Estimator object to save.
    model_name : str
        Name of model.
    """
    pd.to_pickle(model, f"models/{model_name}.pkl")
