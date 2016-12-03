import sys

import nltk
from nltk.stem.porter import *
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


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

training_data = datasets.load_files(sys.argv[1], encoding="utf-8", decode_error='ignore')
bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),tokenizer=tokenize, stop_words='english')

selector = SelectPercentile(chi2, 25)


print("\nSVM\n")
clf = LinearSVC(penalty="l2",dual=False, C=5.0)

pipe_clf = Pipeline([('vectorizer', bigram_tfidf_vectorizer), ('selector',selector), ('classifier',clf )])

pipe_clf.fit(training_data.data, training_data.target)
joblib.dump(pipe_clf, sys.argv[2])

