import copy
import os
import string
import sys

import nltk
import numpy as np
from nltk.stem.porter import *
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
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
test_data = datasets.load_files(sys.argv[2], encoding="utf-8", decode_error='ignore')


def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-20:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))


print("\n---------------------------------\nTFIDF VECTORIZER\n")

print("Config 3.  Using tfidf vectorizer + stop_words filter   + stemming (porter stemmer)\n----------------------------\n")
bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),tokenizer=tokenize, stop_words='english')


bigram_vectors = bigram_tfidf_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_tfidf_vectorizer.transform(test_data.data)


print(bigram_vectors.shape)

print("\nSVM\n")
clf = LinearSVC()

print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))

print("Config 4 Using tfidf vectorizer + stop_words filter + stemming + penalty L1 \n----------------------------\n")
bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),tokenizer=tokenize, stop_words='english')

bigram_vectors = bigram_tfidf_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_tfidf_vectorizer.transform(test_data.data)


print("\nSVM\n")
clf = LinearSVC(penalty="l1",dual=False)

print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nConfig 5 Using tfidf vectorizer + stop_words filter + stemming + penalty L2 + select Percentile 30 \n----------------------------\n")
bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),tokenizer=tokenize, stop_words='english')


bigram_vectors = bigram_tfidf_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_tfidf_vectorizer.transform(test_data.data)

print(bigram_vectors.shape)
selector = SelectPercentile(chi2, 30)

bigram_vectors = selector.fit_transform(bigram_vectors, training_data.target)
bigram_vectors_test = selector.transform(bigram_vectors_test)

print("After SelectPercentile. No. of features in bigram tfidf vector =" + str(bigram_vectors.shape))


print("\nSVM\n")
clf = LinearSVC(penalty="l2",C=1.0)

print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nConfig 8 Using tfidf vectorizer + stop_words filter + stemming + penalty L2 + select Percentile 30  + C=6.0\n----------------------------\n")
bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),tokenizer=tokenize, stop_words='english')


bigram_vectors = bigram_tfidf_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_tfidf_vectorizer.transform(test_data.data)

print(bigram_vectors.shape)
selector = SelectPercentile(chi2, 30)

bigram_vectors = selector.fit_transform(bigram_vectors, training_data.target)
bigram_vectors_test = selector.transform(bigram_vectors_test)

print("After SelectPercentile. No. of features in bigram tfidf vector =" + str(bigram_vectors.shape))


print("\nSVM\n")
clf = LinearSVC(penalty="l2",dual=False, C=6.0)

print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nConfig 9 Using tfidf vectorizer + stop_words filter + stemming + penalty L2 + select Percentile 25 + C = 5.0 \n----------------------------\n")
bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),tokenizer=tokenize, stop_words='english')


bigram_vectors = bigram_tfidf_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_tfidf_vectorizer.transform(test_data.data)

print(bigram_vectors.shape)
selector = SelectPercentile(chi2, 25)

bigram_vectors = selector.fit_transform(bigram_vectors, training_data.target)
bigram_vectors_test = selector.transform(bigram_vectors_test)

print("After SelectPercentile. No. of features in bigram tfidf vector =" + str(bigram_vectors.shape))


print("\nSVM\n")
clf = LinearSVC(penalty="l2",dual=False, C=5.0)

print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nCount vectorizer\n------------------\n")
print("\nConfig 1\n------------------\n")


bigram_count_vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', stop_words='english')

bigram_vectors = bigram_count_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_count_vectorizer.transform(test_data.data)
print(bigram_vectors.shape)

print("\nSVM\n")
clf = LinearSVC()


print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))




print("\nConfig 2\n------------------\n")


bigram_count_vectorizer = CountVectorizer( tokenizer=tokenize, token_pattern=r'\b\w+\b', stop_words='english')

bigram_vectors = bigram_count_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_count_vectorizer.transform(test_data.data)
print(bigram_vectors.shape)

print("\nSVM\n")
clf = LinearSVC()


print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))




class tw_train:
    data = []
    filenames = []
    target = []
    target_names = []

    def __init__(self):
        self.data = []
        self.target_names = []
        self.target = []

walk_dir = sys.argv[1]
test_dir = sys.argv[2]
print("\nConfig 6 Using tfidf vectorizer  + stop_words filter + stemming + penalty L2 + Strip Headers + remove  punctuation\n----------------------------------------\n")

tw = tw_train()
# Copy targets, ie. categories (which are the folder names) to tw.target
tw.target_names = copy.deepcopy([name for name in os.listdir(walk_dir)])

for (dirpath, dirnames, filenames) in os.walk(walk_dir):
    for filename in filenames:
        tw.filenames.append(filename)
        file_path = os.path.join(dirpath, filename)
        with open(file_path, 'r') as f:

            #Strip Headers
            header_offset = 0
            while f.readline().strip():
                header_offset += 1
            content = f.read()

            # Remove punctuations, make lowercase
            lowers = content.lower()
            no_punctuation = lowers.translate(string.punctuation)

            # Add training data content
            tw.data.append(no_punctuation)


            # Add the category index corresponding to training data
            tw.target.append(tw.target_names.index(os.path.basename(dirpath)))



test_files = []
test_data = []
test_target = []

for (dirpath, dirnames, filenames) in os.walk(test_dir):
    for filename in filenames:
        test_files.append(filename)
        file_path = os.path.join(dirpath, filename)
        with open(file_path, 'r') as f:

                #strip headers
                header_offset = 0
                while f.readline().strip():
                     header_offset += 1

                content = f.read()
                # Make text to lower case, remove punctuations
                lowers = content.lower()
                no_punctuation = lowers.translate(string.punctuation)

                # Add training data content
                test_data.append(no_punctuation)

                test_target.append(tw.target_names.index(os.path.basename(dirpath)))

bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize, token_pattern=r'\b\w+\b', stop_words='english')
bigram_vectors = bigram_tfidf_vectorizer.fit_transform(tw.data)
bigram_vectors_test = bigram_tfidf_vectorizer.transform(test_data)

print("\nbigram\n")
print(bigram_vectors.shape)


print("\nSVM\n")
clf = LinearSVC(penalty="l2",dual=False, C=1)

print("\nbigram\n")
clf.fit(bigram_vectors, tw.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_target, pred, average='macro'))


print("\nConfig 7 Using tfidf vectorizer  + stop_words filter + stemming + penalty L2 + remove  punctuation\n----------------------------------------\n")

# Copy targets, ie. categories (which are the folder names) to tw.target
tw.target_names = copy.deepcopy([name for name in os.listdir(walk_dir)])

for (dirpath, dirnames, filenames) in os.walk(walk_dir):
    for filename in filenames:
        tw.filenames.append(filename)
        file_path = os.path.join(dirpath, filename)
        with open(file_path, 'r') as f:

            content = f.read()

            lowers = content.lower()
            no_punctuation = lowers.translate(string.punctuation)
            # Add training data content

            tw.data.append(no_punctuation)

            # Add the category index corresponding to training data
            tw.target.append(tw.target_names.index(os.path.basename(dirpath)))



test_files = []
test_data = []
test_target = []

for (dirpath, dirnames, filenames) in os.walk(test_dir):
    for filename in filenames:
        test_files.append(filename)
        file_path = os.path.join(dirpath, filename)
        with open(file_path, 'r') as f:


                content = f.read()
                # Make text to lower case

                lowers = content.lower()
                no_punctuation = lowers.translate(string.punctuation)


                # Add training data content
                test_data.append(no_punctuation)  # test_data.append(content)

                test_target.append(tw.target_names.index(os.path.basename(dirpath)))


bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize, token_pattern=r'\b\w+\b', stop_words='english')
bigram_vectors = bigram_tfidf_vectorizer.fit_transform(tw.data)
bigram_vectors_test = bigram_tfidf_vectorizer.transform(test_data)

print("\nbigram\n")
print(bigram_vectors.shape)


print("\nSVM\n")
clf = LinearSVC(penalty="l2",dual=False)

print("\nbigram\n")
clf.fit(bigram_vectors, tw.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_target, pred, average='macro'))


