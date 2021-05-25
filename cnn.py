"""
CNN
"""
import pickle
from sklearn.model_selection import train_test_split
import gensim 
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
from keras.layers import Dense, Embedding, Flatten, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
#-----------------------------------------------#
#1. read file
ds1_real = pd.read_csv('ds1_real.csv', sep=',') #label,cleantweet (real,)
ds1_fake = pd.read_csv('ds1_fake.csv', sep=',')
frames = [ds1_real, ds1_fake]
df = pd.concat(frames)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#print(df.head())
df.cleantweet=df.cleantweet.astype(str)

#2. label coding
label_code = {'fake':0, 'real':1}
df['label_code'] = df['label']
df = df.replace({'label_code':label_code})

#3. Train - test split
X_train, X_test, y_train, y_test = train_test_split(df['cleantweet'], df['label_code'], test_size=0.10, random_state=8)

#Load Pretrained Embedding
model = gensim.models.Word2Vec.load('feature_w2v/w2v')
def word2idx(word):
    return (model.wv.key_to_index[word])
def idx2word(idx):
#    #print("idx2word_adjusted funstion, ids: ",idx)
    return model.wv.index_to_key[idx]
pretrained_weights = model.wv.vectors
vocab_size, emdedding_test_size = pretrained_weights.shape
emb_dim = 64
#-------------create idx for sequence input & pad & same length---------#
inputs = [X_train, X_test]
data = pd.concat(inputs)
sentences = []
for line in data: 
    line = line.replace('\n','').split(' ')
    word_index = []
    for word in line: #1 sentence
        index = word2idx(word)
        word_index.append(index)
    
    word_index = np.asarray(word_index)
    sentences.append(word_index)
    
sentences = np.asarray(sentences)
#pad sequences
padded_docs = pad_sequences(sentences, padding='post')

X_train_pad = padded_docs[:9630]  #total training: 9630 
X_test_pad = padded_docs[9630:]
max_length = padded_docs.shape[1]

#-----------------Define the model: CNN---------------------#
early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
ds1_en_cnn_model = 'layer,filter,batch,acc,p,r,f\n' 

model = Sequential()
model.add(Embedding(vocab_size, emb_dim, input_length=max_length, mask_zero=True, weights=[pretrained_weights], trainable=False))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=8))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train, epochs=64, batch_size=64, validation_split=0.33,callbacks=[early_stop])
#model.save('Models/cnn_l'+str(1)+'_k'+str(32)+'_b'+str(64))
# make class predictions with the model
#predictions =  np.argmax(model.predict(X_test_pad))
predictions = model.predict_classes(X_test_pad)
acc = accuracy_score(y_test, predictions)
precision,recall,fscore,support = score(y_test, predictions,average='macro')
ds1_en_cnn_model += str(1)+','+str(32)+',' + str(64) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)+ '\n'

#--------------2 layers----------------#
model = Sequential()
model.add(Embedding(vocab_size, emb_dim, input_length=max_length, mask_zero=True, weights=[pretrained_weights], trainable=False))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=8))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train, epochs=64, batch_size=64, validation_split=0.33,callbacks=[early_stop])
#model.save('Models/cnn_l'+str(2)+'_k'+str(32)+'_b'+str(64))
# make class predictions with the model
#predictions =  np.argmax(model.predict(X_test_pad))
predictions = model.predict_classes(X_test_pad)
acc = accuracy_score(y_test, predictions)
precision,recall,fscore,support = score(y_test, predictions,average='macro')
ds1_en_cnn_model += str(2)+','+str(32)+',' + str(64) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore) +'\n'

#--------------3 layers----------------#
model = Sequential()
model.add(Embedding(vocab_size, emb_dim, input_length=max_length, mask_zero=True, weights=[pretrained_weights], trainable=False))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=8))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train, epochs=64, batch_size=64, validation_split=0.33,callbacks=[early_stop])
#model.save('Models/cnn_l'+str(2)+'_k'+str(32)+'_b'+str(64))
# make class predictions with the model
#predictions =  np.argmax(model.predict(X_test_pad))
predictions = model.predict_classes(X_test_pad)
acc = accuracy_score(y_test, predictions)
precision,recall,fscore,support = score(y_test, predictions,average='macro')
ds1_en_cnn_model += str(3)+','+str(32)+',' + str(64) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)


with open("Results/ds1_en_cnn_model.csv", "w") as text_file:
    text_file.write(ds1_en_cnn_model)