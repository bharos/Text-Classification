import sys

from sklearn import datasets
import nltk
from nltk.stem.porter import *
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support



def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

clf = joblib.load(sys.argv[1])
test_data = datasets.load_files(sys.argv[2], encoding="utf-8", decode_error='ignore')


pred = clf.predict(test_data.data)
print(precision_recall_fscore_support(test_data.target, pred, average='macro'))
