import sys
import pandas as pd
from sqlalchemy import create_engine

# nltk
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import recall_score, make_scorer, classification_report


class UpperCaseWordsRatioExtractor(BaseEstimator, TransformerMixin):
    def get_upper_case_words_ratio(self, text):
        tokens = tokenize(text)
        if len(tokens) < 1:
            return 0
        return sum([word.isupper() for word in tokens]) / len(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_with_upper_case_ratio = pd.Series(X).apply(self.get_upper_case_words_ratio)
        return pd.DataFrame(X_with_upper_case_ratio)
        

def load_data(database_filepath):
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql("messages_categorized", engine)
    X = df.iloc[:, 0]
    Y = df.iloc[:, 3:]
    categories = Y.columns

    return X, Y, categories


def tokenize(text):
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer() # Initialize lemmatizer
    tokens_cleaned = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

    return tokens_cleaned


def make_pipline(enhanced_pipeline=True):
    if enhanced_pipeline:
        return Pipeline([
            ("features", FeatureUnion([
                ("text_pipeline", Pipeline([
                    ("count_vect", CountVectorizer(tokenizer=tokenize)),
                    ("tfidf", TfidfTransformer())])
                ),
                ("upper_case_word", UpperCaseWordsRatioExtractor())
                ])),
            ("clf", MultiOutputClassifier(RandomForestClassifier()))
            ])

    return Pipeline([
        ("countVect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
        ])


def recall_score_avg(Y_test, Y_pred, average='macro'):
    recall_scores = [recall_score(Y_test.values[:,i], Y_pred[:,i], average='macro') for i in range(Y_pred.shape[1])]
    
    return sum(recall_scores) / len(recall_scores)


def grid_search_best_model(X_train, y_train, pipeline, param_grid):
    recall_scorer = make_scorer(recall_score_avg, greater_is_better=True)
    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, cv=3, scoring=recall_scorer)
    cv.fit(X_train, y_train)
    print(cv.best_params_)

    return cv.best_estimator_


def evaluate_model(model, Y_pred, Y_test, category_names):
    for col_index in range(Y_test.shape[1]):
        scores = classification_report(Y_test.values[:, col_index], Y_pred[:, col_index])
        print(scores)


def save_model(model, model_filepath):
    pd.to_pickle(model, f"{model_filepath}.pkl")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Creating pipeline...')
        pipeline = make_pipline()

        parameters = {
        #'countVect__ngram_range': [(1, 1), (1, 2)],
        'features__text_pipeline__count_vect__max_df': [0.5, 1.0], #, 1.0],
        #'countVect__max_features': [None, 5000, 10000],
        'features__text_pipeline__tfidf__use_idf': [True], #, False],
        'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__max_features': ["auto", None],
        'clf__estimator__min_samples_split': [2, 4, 8]
    }

        print('Searching best estimator...')
        best_model = grid_search_best_model(X_train, Y_train, pipeline, parameters)
        
        Y_pred = best_model.predict(X_test)

        print('Evaluating model...')
        evaluate_model(best_model, Y_pred, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()