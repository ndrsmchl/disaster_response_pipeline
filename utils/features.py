import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .functions import tokenize

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
        