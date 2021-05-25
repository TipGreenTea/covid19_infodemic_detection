"""
LSTM from w2v
Loading the model back:
from tensorflow import keras
model = keras.models.load_model('path/to/location')
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
from keras.layers import Dense, Embedding, Flatten, LSTM, GRU, Bidirectional
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
#-----------------------------------------------#
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
print(padded_docs.shape)
#print(padded_Xtrain[0])
X_train_pad = padded_docs[:9630]
print(X_train_pad.shape)
X_test_pad = padded_docs[9630:]
print(X_test_pad.shape)
max_length = padded_docs.shape[1]

#-----------------Define the model: LSTM---------------------#
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
ep = 64
ds1_en_lstm_model = 'neuron,batch,acc,p,r,f\n' 

neurons = [16, 32]
batchs = [8]
for b in batchs:
    for n in neurons:
        model = Sequential()
        model.add(Embedding(vocab_size, emb_dim, input_length=max_length, mask_zero=True, weights=[pretrained_weights], trainable=False))
        model.add(LSTM(n))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #print(model.summary())
        #3. with validation loss --> history graph  (Learning Curves to Diagnose Machine Learning Model Performance slide)
        history = model.fit(X_train_pad, y_train, epochs=ep, batch_size=b, validation_split=0.33,callbacks=[early_stop])
        #model.save('Models/lstm_n'+str(n)+'_b'+str(b))
        # make class predictions with the model
       
        predictions = model.predict_classes(X_test_pad)
        acc = accuracy_score(y_test, predictions)
        precision,recall,fscore,support = score(y_test, predictions,average='macro')
        #matrix = confusion_matrix(labels_test, predictions)
        #report = classification_report(labels_test, predictions)
        #print(acc)
        #print(precision)
        #print(recall)
        #print(fscore)
        ds1_en_lstm_model += str(n)+',' + str(b) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)+ '\n'
        #break
with open("Results/ds1_en_lstm_model.csv", "w") as text_file:
    text_file.write(ds1_en_lstm_model)

    

#-----------------Define the model: GRU---------------------#
#early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
ds1_en_gru_model = 'neuron,batch,acc,p,r,f\n' 

neurons = [16, 32]
batchs = [8]
for b in batchs:
    for n in neurons:
        model = Sequential()
        model.add(Embedding(vocab_size, emb_dim, input_length=max_length, mask_zero=True, weights=[pretrained_weights], trainable=False))
        model.add(GRU(n))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #print(model.summary())
        #3. with validation loss --> history graph  (Learning Curves to Diagnose Machine Learning Model Performance slide)
        history = model.fit(X_train_pad, y_train, epochs=ep, batch_size=b, validation_split=0.33,callbacks=[early_stop])
        #model.save('Models/gru_n'+str(n)+'_b'+str(b))
        # make class predictions with the model
        #predictions =  np.argmax(model.predict(X_test_pad))
        predictions = model.predict_classes(X_test_pad)
        acc = accuracy_score(y_test, predictions)
        precision,recall,fscore,support = score(y_test, predictions,average='macro')
        #matrix = confusion_matrix(labels_test, predictions)
        #report = classification_report(labels_test, predictions)
        #print(acc)
        #print(precision)
        #print(recall)
        #print(fscore)
        ds1_en_gru_model += str(n)+',' + str(b) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)+ '\n'
        #break
with open("Results/ds1_en_gru_model.csv", "w") as text_file:
    text_file.write(ds1_en_gru_model)


#-----------------Define the model: Bi-LSTM---------------------#
#early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
ep = 1
ds1_en_bilstm_model = 'neuron,batch,acc,p,r,f\n' 

neurons = [16, 32]
batchs = [8]
for b in batchs:
    for n in neurons:
        model = Sequential()
        model.add(Embedding(vocab_size, emb_dim, input_length=max_length, mask_zero=True, weights=[pretrained_weights], trainable=False))
        model.add(Bidirectional(LSTM(n)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #print(model.summary())
        #3. with validation loss --> history graph  (Learning Curves to Diagnose Machine Learning Model Performance slide)
        history = model.fit(X_train_pad, y_train, epochs=ep, batch_size=b, validation_split=0.33,callbacks=[early_stop])
        #model.save('Models/lstm_n'+str(n)+'_b'+str(b))
        # make class predictions with the model
       
        predictions = model.predict_classes(X_test_pad)
        acc = accuracy_score(y_test, predictions)
        precision,recall,fscore,support = score(y_test, predictions,average='macro')
        #matrix = confusion_matrix(labels_test, predictions)
        #report = classification_report(labels_test, predictions)
        #print(acc)
        #print(precision)
        #print(recall)
        #print(fscore)
        ds1_en_bilstm_model += str(n)+',' + str(b) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)+ '\n'
        #break
with open("Results/ds1_en_bilstm_model.csv", "w") as text_file:
    text_file.write(ds1_en_bilstm_model)
    
    
#-----------------Define the model: Bi-GRU---------------------#
#early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
ep = 1
ds1_en_bigru_model = 'neuron,batch,acc,p,r,f\n' 

neurons = [16, 32]
batchs = [8]
for b in batchs:
    for n in neurons:
        model = Sequential()
        model.add(Embedding(vocab_size, emb_dim, input_length=max_length, mask_zero=True, weights=[pretrained_weights], trainable=False))
        model.add(Bidirectional(GRU(n)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #print(model.summary())
        #3. with validation loss --> history graph  (Learning Curves to Diagnose Machine Learning Model Performance slide)
        history = model.fit(X_train_pad, y_train, epochs=ep, batch_size=b, validation_split=0.33,callbacks=[early_stop])
        #model.save('Models/lstm_n'+str(n)+'_b'+str(b))
        # make class predictions with the model
       
        predictions = model.predict_classes(X_test_pad)
        acc = accuracy_score(y_test, predictions)
        precision,recall,fscore,support = score(y_test, predictions,average='macro')
        #matrix = confusion_matrix(labels_test, predictions)
        #report = classification_report(labels_test, predictions)
        #print(acc)
        #print(precision)
        #print(recall)
        #print(fscore)
        ds1_en_bigru_model += str(n)+',' + str(b) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)+ '\n'
        #break
with open("Results/ds1_en_bigru_model.csv", "w") as text_file:
    text_file.write(ds1_en_bigru_model)