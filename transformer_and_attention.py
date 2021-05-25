"""
Transformer
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
from keras.layers import Dense, Embedding, Flatten, LSTM, GRU, Bidirectional, Input, Concatenate
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



#Implement a Transformer block as a layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
#Implement embedding layer
#Two seperate embedding layers, one for tokens, one for token index (positions).
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=maxlen, mask_zero=True, weights=[pretrained_weights], trainable=False)
        #model.add(Embedding(vocab_size, emb_dim, input_length=maxlen, mask_zero=True, weights=[pretrained_weights], trainable=False))
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emb_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
#prepare dataset
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
print(y_test.value_counts())
print(y_train.value_counts())
#print(X_train[:5])
print(len(X_train))
#print(X_train[0])
print(len(X_test))


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

#Create classifier model using transformer layer
#Transformer layer outputs one vector for each time step of our input sequence. 
#Here, we take the mean across all time steps and use a feed forward network on top of it to classify text.

embed_dim = 64  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

#Train and Evaluate
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
ep = 32
#b = 8

inputs = layers.Input(shape=(max_length,))
embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


ds1_en_transformer_model = 'head,ff_dim,batch,acc,p,r,f\n' 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_pad, y_train, epochs=ep, batch_size=8, validation_split=0.33,callbacks=[early_stop])
predictions = (model.predict(X_test_pad) > 0.5).astype(int)
#predictions = model.predict_classes(X_test_pad)



acc = accuracy_score(y_test, predictions)
precision,recall,fscore,support = score(y_test, predictions,average='macro')
ds1_en_transformer_model += str(2)+',' + str(32)+',' + str(8) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)+ '\n'

#print('stop')

history = model.fit(X_train_pad, y_train, epochs=ep, batch_size=32, validation_split=0.33,callbacks=[early_stop])
predictions = (model.predict(X_test_pad) > 0.5).astype(int)
acc = accuracy_score(y_test, predictions)
precision,recall,fscore,support = score(y_test, predictions,average='macro')
ds1_en_transformer_model += str(2)+',' + str(32)+',' + str(32) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)+ '\n'

with open("Results/ds1_en_transformer_model.csv", "w") as text_file:
    text_file.write(ds1_en_transformer_model)


#-------------------Bi-RNN with attention-----------------#
ds1_en_birnn_attention_model = 'rnn_type,att_size,rnn_size,batch,acc,p,r,f\n' 

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

#LSTM
sequence_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences = Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_length, mask_zero=True, weights=[pretrained_weights], trainable=False)(sequence_input)
#Bi-directional LSTM
#lstm = Bidirectional(LSTM(16, dropout=0.3,return_sequences=True, return_state=True, recurrent_activation='relu',recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(embedded_sequences)
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(16,dropout=0.2,return_sequences=True,return_state=True,recurrent_activation='relu',recurrent_initializer='glorot_uniform'))(embedded_sequences)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
attention = Attention(32) #unit

context_vector, attention_weights = attention(lstm, state_h)
output = keras.layers.Dense(1, activation='sigmoid')(context_vector)
model = keras.Model(inputs=sequence_input, outputs=output)

# summarize layers
#print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train, epochs=ep, batch_size=8, validation_split=0.33,callbacks=[early_stop])
predictions = (model.predict(X_test_pad) > 0.5).astype(int)
acc = accuracy_score(y_test, predictions)
precision,recall,fscore,support = score(y_test, predictions,average='macro')
ds1_en_birnn_attention_model += 'bi-lstm,'+str(32)+',' + str(16)+',' + str(8) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)+ '\n'




#GRU
sequence_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences = Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_length, mask_zero=True, weights=[pretrained_weights], trainable=False)(sequence_input)
#Bi-directional LSTM
#lstm = Bidirectional(LSTM(16, dropout=0.3,return_sequences=True, return_state=True, recurrent_activation='relu',recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(embedded_sequences)
lstm, forward_h,  backward_h = Bidirectional(GRU(16,dropout=0.2,return_sequences=True,return_state=True,recurrent_activation='relu',recurrent_initializer='glorot_uniform'))(embedded_sequences)
state_h = Concatenate()([forward_h, backward_h])
#state_c = Concatenate()([forward_c, backward_c])
attention = Attention(32) #unit
context_vector, attention_weights = attention(lstm, state_h)
output = keras.layers.Dense(1, activation='sigmoid')(context_vector)
model = keras.Model(inputs=sequence_input, outputs=output)

# summarize layers
#print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_pad, y_train, epochs=ep, batch_size=8, validation_split=0.33,callbacks=[early_stop])
predictions = (model.predict(X_test_pad) > 0.5).astype(int)
acc = accuracy_score(y_test, predictions)
precision,recall,fscore,support = score(y_test, predictions,average='macro')

ds1_en_birnn_attention_model += 'bi-gru,'+str(32)+',' + str(16)+',' + str(8) +',' +str(acc)+','+str(precision)+','+str(recall) +','+ str(fscore)+ '\n'

with open("Results/ds1_en_birnn_attention_model.csv", "w") as text_file:
    text_file.write(ds1_en_birnn_attention_model)
