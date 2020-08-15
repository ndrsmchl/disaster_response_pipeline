# nltk
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def tokenize(text):
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer() # Initialize lemmatizer
    tokens_cleaned = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

    return tokens_cleaned