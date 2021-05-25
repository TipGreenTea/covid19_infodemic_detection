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

# some time later...
# load the model from disk
model_log = load(open('Models/ds1_en_model_log.sav', 'rb')) 
model_lda = load(open('Models/ds1_en_model_lda.sav', 'rb'))
model_knn = load(open('Models/ds1_en_model_knn.sav', 'rb'))
model_nb = load(open('Models/ds1_en_model_nb.sav', 'rb'))
model_dt = load(open('Models/ds1_en_model_dt.sav', 'rb'))
model_svc = load(open('Models/ds1_en_model_svc.sav', 'rb'))
model_rf = load(open('Models/ds1_en_model_rf.sav', 'rb'))
      
   
#predict
predicted_log = model_log.predict(features_test)
predicted_lda = model_lda.predict(features_test)
predicted_knn = model_knn.predict(features_test)
predicted_nb = model_nb.predict(features_test)
predicted_dt = model_dt.predict(features_test)
predicted_svc = model_svc.predict(features_test)
predicted_rf = model_rf.predict(features_test)

#-----------------------Evaluation Results--------------------------#

# Test accuracy
print(accuracy_score(labels_test, predicted_log))
print(accuracy_score(labels_test, predicted_lda))
print(accuracy_score(labels_test, predicted_knn))
print(accuracy_score(labels_test, predicted_nb))
print(accuracy_score(labels_test, predicted_dt))
print(accuracy_score(labels_test, predicted_svc))
print(accuracy_score(labels_test, predicted_rf))
ds1_en_ml_acc = 'model,acc\n' 
ds1_en_ml_acc += 'log,' + str(accuracy_score(labels_test, predicted_log)) +'\n'
ds1_en_ml_acc += 'lda,' + str(accuracy_score(labels_test, predicted_lda)) +'\n'
ds1_en_ml_acc += 'knn,' + str(accuracy_score(labels_test, predicted_knn)) +'\n'
ds1_en_ml_acc += 'nb,' + str(accuracy_score(labels_test, predicted_nb)) +'\n'
ds1_en_ml_acc += 'dt,' + str(accuracy_score(labels_test, predicted_dt)) +'\n'
ds1_en_ml_acc += 'svc,' + str(accuracy_score(labels_test, predicted_svc)) +'\n'
ds1_en_ml_acc += 'rf,' + str(accuracy_score(labels_test, predicted_rf))
with open("Results/ds1_en_ml_acc.csv", "w") as text_file:
    text_file.write(ds1_en_ml_acc)

#precision_recall_fscore_support
precision_log,recall_log,fscore_log,support_log = score(labels_test, predicted_log,average='macro')
precision_lda,recall_lda,fscore_lda,support_lda = score(labels_test, predicted_lda,average='macro')
precision_knn,recall_knn,fscore_knn,support_knn = score(labels_test, predicted_knn,average='macro')
precision_nb,recall_nb,fscore_nb,support_nb = score(labels_test, predicted_nb,average='macro')
precision_dt,recall_dt,fscore_dt,support_dt = score(labels_test, predicted_dt,average='macro')
precision_svc,recall_svc,fscore_svc,support_svc = score(labels_test, predicted_svc,average='macro')
precision_rf,recall_rf,fscore_rf,support_rf = score(labels_test, predicted_rf,average='macro')

ds1_en_ml_prf = 'model,p,r,f\n' 
ds1_en_ml_prf += 'log,' + str(precision_log) +','+ str(recall_log) +','+ str(fscore_log) +'\n'
ds1_en_ml_prf += 'lda,' + str(precision_lda) +','+ str(recall_lda) +','+ str(fscore_lda) +'\n'
ds1_en_ml_prf += 'knn,' + str(precision_knn) +','+ str(recall_knn) +','+ str(fscore_knn) +'\n'
ds1_en_ml_prf += 'nb,' + str(precision_nb) +','+ str(recall_nb) +','+ str(fscore_nb) +'\n'
ds1_en_ml_prf += 'dt,' + str(precision_dt) +','+ str(recall_dt) +','+ str(fscore_dt) +'\n'
ds1_en_ml_prf += 'svc,' + str(precision_svc) +','+ str(recall_svc) +','+ str(fscore_svc) +'\n'
ds1_en_ml_prf += 'rf,' + str(precision_rf) +','+ str(recall_rf) +','+ str(fscore_rf)
with open("Results/ds1_en_ml_prf.csv", "w") as text_file:
    text_file.write(ds1_en_ml_prf)

# Classification Report (P,R,F)
#report_log = classification_report(labels_test, predicted_log)
#print(report_log)



# Confusion Matrix
matrix_log = confusion_matrix(labels_test, predicted_svc)
print(matrix_log)


# Custom Confusion Matrix 
aux_df = df[['label', 'label_code']].drop_duplicates().sort_values('label_code')
plt.figure(figsize=(12.8,6))
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
sns.set(font_scale=3.6)
sns.heatmap(matrix_log, 
            annot=True,
            xticklabels=aux_df['label'].values, 
            yticklabels=aux_df['label'].values,
            cmap="Blues",
            fmt='g')
plt.ylabel('Actual',  fontsize=26)
plt.xlabel('Predicted', fontsize=26)
#plt.title('Confusion matrix')
plt.ticklabel_format(useOffset=False)
plt.show()
plt.savefig('pic/ds1_en_svm_confusionmatrix.png')
