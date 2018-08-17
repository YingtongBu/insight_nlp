import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

'''
corpus
'''
train_corpus = []
train_label = []
with open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/train_project.txt', 'r') as file_object:
    for line in file_object.readlines():
        train_corpus.append(line.rstrip('\n').split('\t')[0])
        train_label.append(line.rstrip('\n').split('\t')[1])

test_corpus = []
test_label = []
with open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/test_project.txt', 'r') as file_object:
    for line in file_object.readlines():
        test_corpus.append(line.rstrip('\n').split('\t')[0])
        test_label.append(line.rstrip('\n').split('\t')[1])

'''
train
'''
vectorizer = CountVectorizer(ngram_range=(2, 4))
X_train = vectorizer.fit_transform(train_corpus)

y_train = np.asarray(train_label)
print(X_train)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5), algorithm="SAMME.R", n_estimators=500, learning_rate=0.8)
bdt.fit(X_train, y_train)

'''
test
'''
X_test = vectorizer.transform(test_corpus)
y_test = np.asarray(test_label)
y_predict = bdt.predict(X_test)
y_true = y_test
print(f1_score(y_true, y_predict, average = None))