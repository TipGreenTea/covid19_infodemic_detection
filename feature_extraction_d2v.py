"""
2. Doc2Vec
"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import string
import matplotlib.pyplot as plt
from google_trans_new import google_translator   
import warnings
warnings.filterwarnings('ignore')
from pythainlp import sent_tokenize, word_tokenize
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

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

#--------------------D2V----------------------#
inputs = [X_train, X_test]
data = pd.concat(inputs)


#Here we have a list of four sentences as training data. 
#Now I have tagged the data and its ready for training. 
#Lets start training our model.
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
#print("tagged_data: ",tagged_data)
max_epochs = 64#64
vec_size = 200#128
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,alpha=alpha, min_alpha=0.00025, min_count=1, dm =1)
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,total_examples=model.corpus_count,epochs=model.epochs)
    model.alpha -= 0.0002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no decay

# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
features_train = []
for row in X_train: 
    v = word_tokenize(row.lower())
    v1 = model.infer_vector(v)
    features_train.append(v1)
features_train = np.asarray(features_train)
print(features_train.shape)
labels_train = y_train

features_test = []
for row in X_test: 
    v = word_tokenize(row.lower())
    v1 = model.infer_vector(v)
    features_test.append(v1)
features_test = np.asarray(features_test)
print(features_test.shape)
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
    plt.savefig('pic/d2v_'+model+'_'+str(vec_size)+'.png')
    
plot_dim_red("PCA", features=features, labels=labels, n_components=2)
plot_dim_red("TSNE", features=features, labels=labels, n_components=2)

#--------------------Feature Save----------------------#
#6. Save to use them later
# X_train
with open('feature_d2v/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)
    
# X_test    
with open('feature_d2v/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
    
# y_train
with open('feature_d2v/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
    
# y_test
with open('feature_d2v/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)
    
# df
with open('feature_d2v/df.pickle', 'wb') as output:
    pickle.dump(df, output)
    
# features_train
with open('feature_d2v/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('feature_d2v/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('feature_d2v/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('feature_d2v/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# D2V object
model.save("feature_d2v/d2v.model")
print("Doc2Vec Model Saved")
