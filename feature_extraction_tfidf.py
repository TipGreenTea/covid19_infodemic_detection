"""
1. TF-IDF
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

#-----------------------------------------------#
ds1_real = pd.read_csv('ds1_real.csv', sep=',') #label,cleantweet (real,)
ds1_fake = pd.read_csv('ds1_fake.csv', sep=',')
frames = [ds1_real, ds1_fake]
df = pd.concat(frames)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#print(df.head())

#2. label coding
label_code = {'fake':0, 'real':1}
df['label_code'] = df['label']
df = df.replace({'label_code':label_code})


#3. Train - test split
X_train, X_test, y_train, y_test = train_test_split(df['cleantweet'], df['label_code'], test_size=0.10, random_state=8)
print(y_test.value_counts())
print(y_train.value_counts())

#--------------------TF-IDF----------------------#
# 4. Text representation
# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

features_train = tfidf.fit_transform(X_train.values.astype('U')).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test.values.astype('U')).toarray()
labels_test = y_test
print(features_test.shape)


#--------------------Feature Visualization----------------------#
#5. Dimensionality Reduction Plots
#Let's do the concatenation:
features = np.concatenate((features_train,features_test), axis=0)
labels = np.concatenate((labels_train,labels_test), axis=0)
print(features.shape)
print(labels.shape)

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
    sns.scatterplot(x='PC1',y='PC2',hue="label_name", data=df_full,
                    palette=["royalblue", "pink"], #"red", "greenyellow", "lightseagreen"
                    alpha=.7).set_title(title);
    plt.savefig('pic/tfidf_'+model+'_'+str(max_features)+'.png')
    
plot_dim_red("PCA", features=features, labels=labels, n_components=2)
plot_dim_red("TSNE", features=features, labels=labels, n_components=2)


#--------------------Feature Save----------------------#
#6. Save to use them later
# X_train
with open('feature_tfidf/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)
    
# X_test    
with open('feature_tfidf/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
    
# y_train
with open('feature_tfidf/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
    
# y_test
with open('feature_tfidf/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)
    
# df
with open('feature_tfidf/df.pickle', 'wb') as output:
    pickle.dump(df, output)
    
# features_train
with open('feature_tfidf/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('feature_tfidf/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('feature_tfidf/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('feature_tfidf/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# TF-IDF object
with open('feature_tfidf/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)
