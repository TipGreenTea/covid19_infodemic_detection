"""
3. W2V

"""
import gzip
import gensim 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import json
import matplotlib.pyplot as plt
from pythainlp import sent_tokenize, word_tokenize
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE

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

#--------------------w2v----------------------#
INPUT_DIM = 64
num_epoch = 32
'''
inputs = [X_train, X_test]
data = pd.concat(inputs)
sentences = []
index = 0 
for line in data: 
    line = line.replace('\n','').split(' ')
    #line = line[:-1] 
    print(line)
    sentences.append(line)
    index +=1
#print(sentences.shape)

model = gensim.models.Word2Vec(sentences, vector_size=INPUT_DIM, window=15, min_count=1, workers=10)
model.train(sentences,total_examples=len(sentences),epochs=num_epoch)
##------------------Save--------------------##
model.save('feature_w2v/w2v')
print("End of Word Embedding Training...")
'''
#---------------------Avg words embedding to represent document--------------------#
#Load Pretrained Embedding
model = gensim.models.Word2Vec.load('feature_w2v/w2v')
def word2idx(word):
    return (model.wv.key_to_index[word])
def idx2word(idx):
#    #print("idx2word_adjusted funstion, ids: ",idx)
    return model.wv.index_to_key[idx]

pretrained_weights = model.wv.vectors

sentences = []
for line in X_train: 
    line = line.replace('\n','').split(' ')
    word_weight = []
    for word in line: #1 sentence
        index = word2idx(word)
        word_weight.append(pretrained_weights[index])
    
    #print(word_weight)
    word_weight = np.asarray(word_weight)
    sentence = np.mean(word_weight, axis=0)
    #print(sentence)
    sentences.append(sentence)
    
features_train = np.asarray(sentences)
#print(features_train.shape)
labels_train = y_train

sentences = []
for line in X_test: 
    line = line.replace('\n','').split(' ')
    word_weight = []
    for word in line: #1 sentence
        index = word2idx(word)
        word_weight.append(pretrained_weights[index])
    
    #print(word_weight)
    word_weight = np.asarray(word_weight)
    sentence = np.mean(word_weight, axis=0)
    #print(sentence)
    sentences.append(sentence)
    
features_test = np.asarray(sentences)
#print(features_train.shape)
labels_test = y_test

#--------------------Feature Visualization----------------------#
#5. Dimensionality Reduction Plots
#Let's do the concatenation:
features = np.concatenate((features_train,features_test), axis=0)
labels = np.concatenate((labels_train,labels_test), axis=0)

def plot_dim_red(model, features, labels, n_components=2):
    # Creation of the model
    if (model == 'PCA'):
        mod = PCA(n_components=n_components)
        title = "PCA decomposition"  # for the plot
    elif (model == 'TSNE'):
        mod = TSNE(n_components=2)
        title = "t-SNE decomposition" 
    else:
        return "Error"
    
    # Fit and transform the features
    principal_components = mod.fit_transform(features)
    # Put them into a dataframe
    df_features = pd.DataFrame(data=principal_components,
                     columns=['PC1', 'PC2'])
    # Now we have to paste each row's label and its meaning
    # Convert labels array to df
    df_labels = pd.DataFrame(data=labels,columns=['label'])
    df_full = pd.concat([df_features, df_labels], axis=1)
    df_full['label'] = df_full['label'].astype(str)
    # Get labels name
    category_names = {
        "0": 'fake',
        "1": 'real'
    }
    # And map labels
    df_full['label_name'] = df_full['label']
    df_full = df_full.replace({'label_name':category_names})
    # Plot
    plt.figure(figsize=(10,10))
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.scatterplot(x='PC1',y='PC2',hue="label_name", data=df_full,
                    palette=["royalblue", "pink"], #"red", "greenyellow", "lightseagreen"
                    alpha=.7).set_title(title);
    plt.savefig('pic/w2v_'+model+'_'+str(INPUT_DIM)+'.png')
    
plot_dim_red("PCA", features=features, labels=labels, n_components=2)
plot_dim_red("TSNE", features=features, labels=labels, n_components=2)

#--------------------Feature Save----------------------#
#6. Save to use them later
# X_train
with open('feature_w2v/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)
    
# X_test    
with open('feature_w2v/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
    
# y_train
with open('feature_w2v/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
    
# y_test
with open('feature_w2v/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)
    
# df
with open('feature_w2v/df.pickle', 'wb') as output:
    pickle.dump(df, output)
    
# features_train
with open('feature_w2v/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('feature_w2v/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('feature_w2v/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('feature_w2v/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    


    
    

