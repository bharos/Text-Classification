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

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
        # count = Counter(stemmed)
        # print(count.most_common(10))
    return stemmed

def tokenize(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)

    return stems

test_dir = "Test"
test_files = []
test_data = []
test_target = []

#Load model from pickle
clf = pickle.load( open( "model.p", "rb" ) )

target_names = copy.deepcopy([name for name in os.listdir(test_dir)])

for (dirpath, dirnames, filenames) in os.walk(test_dir):
    for filename in filenames:
        test_files.append(filename)
        file_path = os.path.join(dirpath, filename)
        with open(file_path, 'r') as f:
            try:
                header_offset = 0
                while f.readline().strip():
                    header_offset += 1

                content = f.read()
                # Make text to lower case

                lowers = content.lower()
                no_punctuation = lowers.translate(string.punctuation)
                #                token_dict[file] = no_punctuation

                # Add training data content
                test_data.append(no_punctuation)
                #test_data.append(content)


                test_target.append(target_names.index(os.path.basename(dirpath)))
            except Exception as e:
                print("Error occured : " + str(e))
print("len = ", len(test_data))
#
# X_new_counts = count_vect.transform(test_data)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# predicted = clf.predict(X_new_tfidf)
predicted = clf.predict(test_data)
print(np.mean(predicted == test_target))

print(f1_score(test_target, predicted, average='macro'))

# for doc, category in zip(test_data, predicted):
#     print("\n".join(doc.split("\n")[:3]))
#     print(tw.target_names[category])
#     print("---------------------------------------------")

