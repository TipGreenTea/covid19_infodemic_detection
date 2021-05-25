import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pickle import dump
from pickle import load

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#First, we load the data:
with open('feature_w2v/df.pickle', 'rb') as data:
    df = pickle.load(data)
# features_train
with open('feature_w2v/features_train.pickle', 'rb') as data:
    features_train = pickle.load(data)
# labels_train
with open('feature_w2v/labels_train.pickle', 'rb') as data:
    labels_train = pickle.load(data)
# features_test
with open('feature_w2v/features_test.pickle', 'rb') as data:
    features_test = pickle.load(data)
# labels_test
with open('feature_w2v/labels_test.pickle', 'rb') as data:
    labels_test = pickle.load(data)
print(features_train.shape)
print(features_test.shape)

#Models
model_log = LogisticRegression(solver='liblinear')
model_lda = LinearDiscriminantAnalysis()
model_knn = KNeighborsClassifier()
model_nb = GaussianNB()
model_dt = DecisionTreeClassifier()
model_svc = SVC()
model_rf = RandomForestClassifier(n_estimators=100, max_features=3)

#Train
model_log.fit(features_train, labels_train)
model_lda.fit(features_train, labels_train)
model_knn.fit(features_train, labels_train)
model_nb.fit(features_train, labels_train)
model_dt.fit(features_train, labels_train)
model_svc.fit(features_train, labels_train)
model_rf.fit(features_train, labels_train)


#Save and Load Machine Learning Models
# save the model to disk
dump(model_log, open('Models/ds1_en_model_log.sav', 'wb'))
dump(model_lda, open('Models/ds1_en_model_lda.sav', 'wb'))
dump(model_knn, open('Models/ds1_en_model_knn.sav', 'wb'))
dump(model_nb, open('Models/ds1_en_model_nb.sav', 'wb'))
dump(model_dt, open('Models/ds1_en_model_dt.sav', 'wb'))
dump(model_svc, open('Models/ds1_en_model_svc.sav', 'wb'))
dump(model_rf, open('Models/ds1_en_model_rf.sav', 'wb'))

# load the model from disk
#loaded_model = load(open(filename, 'rb')) 
#result = loaded_model.score(X_test, Y_test) 
#print(result)


