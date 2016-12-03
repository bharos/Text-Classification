import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.stem.porter import *
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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





print(sys.argv)
training_data = datasets.load_files(sys.argv[1], encoding="utf-8", decode_error='ignore')
test_data = datasets.load_files(sys.argv[2], encoding="utf-8", decode_error='ignore')


def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-20:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))





print("\n---------------------------------\nTFIDF VECTORIZER\n")


unigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))


unigram_vectors = unigram_tfidf_vectorizer.fit_transform(training_data.data)
unigram_vectors_test = unigram_tfidf_vectorizer.transform(test_data.data)

bigram_vectors = bigram_tfidf_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_tfidf_vectorizer.transform(test_data.data)

print(unigram_vectors.shape)
print(bigram_vectors.shape)


print("\nNaive Bayes\n")

split_size = 30
score_array = []
lengths = []
print(len(training_data.data))
print(len(training_data.target))
for i in range(split_size):
    this_loop_len = int(((i + 1) / split_size) * len(training_data.data))
    lengths.append(this_loop_len)
    unigram_vectors = unigram_tfidf_vectorizer.fit_transform(training_data.data[:this_loop_len])
    unigram_vectors_test = unigram_tfidf_vectorizer.transform(test_data.data)
    clf = MultinomialNB()


    print("\nunigram\n")
    clf.fit(unigram_vectors, training_data.target[:this_loop_len])
    pred = clf.predict(unigram_vectors_test)
    score_array.append(metrics.f1_score(test_data.target, pred, average='macro'))
    print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))

plt.plot(lengths,score_array,c='red',lw=2,label="Naive Bayes")

red_patch = mpatches.Patch(color='red', label='Naive Bayes')
print(lengths)
print(score_array)


bigram_vectors = bigram_tfidf_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_tfidf_vectorizer.transform(test_data.data)
print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nLogistic Regression\n------\n")

score_array = []
lengths = []
print(len(training_data.data))
print(len(training_data.target))
for i in range(split_size):
    this_loop_len = int(((i + 1) / split_size) * len(training_data.data))
    lengths.append(this_loop_len)
    unigram_vectors = unigram_tfidf_vectorizer.fit_transform(training_data.data[:this_loop_len])
    unigram_vectors_test = unigram_tfidf_vectorizer.transform(test_data.data)
    clf = LogisticRegression()

    print("\nunigram\n")
    clf.fit(unigram_vectors, training_data.target[:this_loop_len])
    pred = clf.predict(unigram_vectors_test)
    score_array.append(metrics.f1_score(test_data.target, pred, average='macro'))
    print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))

plt.plot(lengths,score_array,c='blue',lw=2, label="Log_Reg")


blue_patch = mpatches.Patch(color='blue', label='Logistic Regression')
plt.legend(handles=[red_patch, blue_patch])
print(lengths)
print(score_array)


print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))



print("\nSVM\n")
score_array = []
lengths = []
print(len(training_data.data))
print(len(training_data.target))
for i in range(split_size):
    this_loop_len = int(((i + 1) / split_size) * len(training_data.data))
    lengths.append(this_loop_len)
    unigram_vectors = unigram_tfidf_vectorizer.fit_transform(training_data.data[:this_loop_len])
    unigram_vectors_test = unigram_tfidf_vectorizer.transform(test_data.data)
    clf = LinearSVC()

    print("\nunigram\n")
    clf.fit(unigram_vectors, training_data.target[:this_loop_len])
    pred = clf.predict(unigram_vectors_test)
    score_array.append(metrics.f1_score(test_data.target, pred, average='macro'))
    print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))

plt.plot(lengths,score_array,c='green', lw=2, label="SVM")


green_patch = mpatches.Patch(color='green', label='SVM')

print(lengths)
print(score_array)



print("\nRandom Forest\n------\n")


score_array = []
lengths = []
print(len(training_data.data))
print(len(training_data.target))

for i in range(split_size):

    this_loop_len = int(((i+1)/split_size)*len(training_data.data))
    lengths.append(this_loop_len)
    unigram_vectors = unigram_tfidf_vectorizer.fit_transform(training_data.data[:this_loop_len])
    unigram_vectors_test = unigram_tfidf_vectorizer.transform(test_data.data)

    clf = RandomForestClassifier()
    print("\nunigram\n")
    clf.fit(unigram_vectors, training_data.target[:this_loop_len])
    pred = clf.predict(unigram_vectors_test)
    score_array.append(metrics.f1_score(test_data.target, pred, average='macro'))
    print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))

plt.plot(lengths,score_array,c='black',lw=2, label="Rand_Forest")
plt.ylabel('F1 Score')
plt.xlabel('Training Data Size')

black_patch = mpatches.Patch(color='black', label='Random Forest')

print(lengths)
print(score_array)
# plt.legend(bbox_to_anchor=(1, 1),
#            bbox_transform=plt.gcf().transFigure,handles=[red_patch, blue_patch, green_patch, black_patch])
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
fig = plt.figure()

plt.show()



print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nCount vectorizer\n------------------\n")


unigram_count_vectorizer = CountVectorizer(ngram_range = (1,1))
bigram_count_vectorizer = CountVectorizer(ngram_range = (1,2), token_pattern=r'\b\w+\b', min_df=1)

unigram_vectors = unigram_count_vectorizer.fit_transform(training_data.data)
unigram_vectors_test = unigram_count_vectorizer.transform(test_data.data)

bigram_vectors = bigram_count_vectorizer.fit_transform(training_data.data)
bigram_vectors_test = bigram_count_vectorizer.transform(test_data.data)

print(unigram_vectors.shape)
print(bigram_vectors.shape)


print("\nNaive Bayes\n")

clf = MultinomialNB()
print("\nunigram\n")
clf.fit(unigram_vectors, training_data.target)
pred = clf.predict(unigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nLogistic Regression\n------\n")

clf = LogisticRegression()
print("\nunigram\n")
clf.fit(unigram_vectors, training_data.target)
pred = clf.predict(unigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))

print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nSVM\n")
clf = LinearSVC()

print("\nunigram\n")
clf.fit(unigram_vectors, training_data.target)
pred = clf.predict(unigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))

print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nRandom Forest\n------\n")
clf = RandomForestClassifier()
print("\nunigram\n")
clf.fit(unigram_vectors, training_data.target)
pred = clf.predict(unigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))


print("\nbigram\n")
clf.fit(bigram_vectors, training_data.target)
pred = clf.predict(bigram_vectors_test)
print(metrics.f1_score(test_data.target, pred, average='macro'))
print(metrics.precision_recall_fscore_support(test_data.target, pred, average='macro'))

