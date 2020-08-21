
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
#from .functions import tokenize

class LengthExtractor(BaseEstimator, TransformerMixin):
    def get_length(self, text):
        return len(text)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_w_length = pd.Series(X).apply(self.get_length)
        return pd.DataFrame(X_w_length)
        