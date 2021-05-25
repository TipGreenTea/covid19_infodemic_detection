"""
ANN from 3 features (if-idf, d2v, w2v)
"""
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pickle import dump
from pickle import load
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

#-----Please change pre-trained features if you want to use different extraction methods----#
#--------------W2V-----------------#
print('#--------------W2V-----------------#')
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


#---------------ANN----------------#
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model = Sequential()
model.add(Dense(32, input_dim=features_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


#3. with validation loss --> history graph  (Learning Curves to Diagnose Machine Learning Model Performance slide)
history = model.fit(features_train, labels_train, epochs=64, batch_size=8, validation_split=0.33,callbacks=[early_stop])

#Save and Load Machine Learning Models
# save the model to disk
#dump(model, open('Models/ds1_en_ann_w2v.sav', 'wb'))

# make class predictions with the model
predictions = model.predict_classes(features_test)
acc = accuracy_score(labels_test, predictions)
precision,recall,fscore,support = score(labels_test, predictions,average='macro')


ds1_en_ann_model = 'feature,acc,p,r,f\n' 
ds1_en_ann_model += 'w2v,' + str(acc) +',' +str(precision)+','+str(recall) +','+ str(fscore)+ '\n'
    
#--------------D2V-----------------#
print('#--------------D2V-----------------#')
#First, we load the data:
with open('feature_d2v/df.pickle', 'rb') as data:
    df = pickle.load(data)
# features_train
with open('feature_d2v/features_train.pickle', 'rb') as data:
    features_train = pickle.load(data)
# labels_train
with open('feature_d2v/labels_train.pickle', 'rb') as data:
    labels_train = pickle.load(data)
# features_test
with open('feature_d2v/features_test.pickle', 'rb') as data:
    features_test = pickle.load(data)
# labels_test
with open('feature_d2v/labels_test.pickle', 'rb') as data:
    labels_test = pickle.load(data)


#---------------ANN----------------#
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model = Sequential()
model.add(Dense(32, input_dim=features_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


#3. with validation loss --> history graph  (Learning Curves to Diagnose Machine Learning Model Performance slide)
history = model.fit(features_train, labels_train, epochs=64, batch_size=8, validation_split=0.33,callbacks=[early_stop])

#Save and Load Machine Learning Models
# save the model to disk
#dump(model, open('Models/ds1_en_ann_d2v.sav', 'wb'))

# make class predictions with the model
predictions = model.predict_classes(features_test)
acc = accuracy_score(labels_test, predictions)
precision,recall,fscore,support = score(labels_test, predictions,average='macro')

ds1_en_ann_model += 'd2v,' + str(acc) +',' +str(precision)+','+str(recall) +','+ str(fscore)+ '\n'

#--------------TFIDF-----------------#
print('#--------------TFIDF-----------------#')
#First, we load the data:
with open('feature_tfidf/df.pickle', 'rb') as data:
    df = pickle.load(data)
# features_train
with open('feature_tfidf/features_train.pickle', 'rb') as data:
    features_train = pickle.load(data)
# labels_train
with open('feature_tfidf/labels_train.pickle', 'rb') as data:
    labels_train = pickle.load(data)
# features_test
with open('feature_tfidf/features_test.pickle', 'rb') as data:
    features_test = pickle.load(data)
# labels_test
with open('feature_tfidf/labels_test.pickle', 'rb') as data:
    labels_test = pickle.load(data)


#---------------ANN----------------#
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model = Sequential()
model.add(Dense(32, input_dim=features_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


#3. with validation loss --> history graph  (Learning Curves to Diagnose Machine Learning Model Performance slide)
history = model.fit(features_train, labels_train, epochs=64, batch_size=8, validation_split=0.33,callbacks=[early_stop])

#Save and Load Machine Learning Models
# save the model to disk
#dump(model, open('Models/ds1_en_ann_tfidf.sav', 'wb'))

# make class predictions with the model
predictions = model.predict_classes(features_test)
acc = accuracy_score(labels_test, predictions)
precision,recall,fscore,support = score(labels_test, predictions,average='macro')

ds1_en_ann_model += 'tfidf,' + str(acc) +',' +str(precision)+','+str(recall) +','+ str(fscore)
    
    
with open("Results/ds1_en_ann_model.csv", "w") as text_file:
    text_file.write(ds1_en_ann_model)