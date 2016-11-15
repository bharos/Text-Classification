import numpy as np
import os
import sys
import copy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter
from nltk.stem.porter import *

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

print('walk_dir = ' + walk_dir)

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
      #  count = Counter(stemmed)
       # print(count.most_common(10))
    return stemmed

def tokenize(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)

    return stems

tw = tw_train()

# Copy targets, ie. categories (which are the folder names) to tw.target
tw.target_names = copy.deepcopy([name for name in os.listdir(walk_dir)])

for (dirpath, dirnames, filenames) in os.walk(walk_dir):
    for filename in filenames:
        tw.filenames.append(filename)
        file_path = os.path.join(dirpath, filename)
        with open(file_path, 'r') as f:
            try:
                header_offset = 0
                while  f.readline().strip():
                    header_offset += 1

                content = f.read()
                #Make text to lower case

                lowers = content.lower()
                no_punctuation = lowers.translate(string.punctuation)


#                token_dict[file] = no_punctuation

                # Add training data content
                tw.data.append(no_punctuation)
 #               tw.data.append(content)

                # Add the category index corresponding to training data
                tw.target.append(tw.target_names.index(os.path.basename(dirpath)))

            except Exception as e:
                print("Error occured : " + str(e))
                exit()

print(tw.target)


# Without Pipeline
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(tw.data)
# print(X_train_counts.shape)
# #print(count_vect.vocabulary_.get(u'algorithm'))
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)
#
# clf = MultinomialNB(alpha=.01)
# clf.fit(X_train_tfidf, tw.target)


# Naive Bayes pipeline
text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
('tfidf', TfidfTransformer()),
('clf', MultinomialNB())])
clf = text_clf.fit(tw.data, tw.target)

# SVM Pipeline
# text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                                            alpha=1e-3, n_iter=5, random_state=42))])


# Logistic Regression Pipeline
# text_clf = Pipeline([('vect', CountVectorizer()),
# ('tfidf', TfidfTransformer()),
# ('clf',  )])

clf = text_clf.fit(tw.data, tw.target)

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-20:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))

print(text_clf.named_steps['clf'])
print(text_clf.named_steps['vect'])tw.target_names)


pickle.dump( clf, open( "model.p", "wb" ) )
