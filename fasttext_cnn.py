# -*- coding: utf-8 -*-
"""bert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LQDUa5ZoJxinKEcNokvaVCcH83PX1ZmF
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import keras
import string
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from nltk.tokenize import sent_tokenize,word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs
from sklearn.model_selection import train_test_split



nltk.download('stopwords')
nltk.download('punkt')

sns.set_style("whitegrid")
np.random.seed(0)

DATA_PATH = '../input/'
EMBEDDING_DIR = '../input/'

MAX_NB_WORDS = 100000
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('/content/drive/MyDrive/ML Assignment/Assignment_6/wiki-news-300d-1M.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

text=pd.read_csv('/content/drive/MyDrive/ML Assignment/Assignment_6/Preprocessed_Final_Datatset_new.csv',encoding='latin1')

text_list=text['Clean_TweetText']
labels=text['label']

X_train, X_test, y_train, y_test = train_test_split(text_list, labels, test_size=0.30, random_state=42)
print(X_train)

print(type(X_train))
train_df=pd.DataFrame()
X_train_list=X_train.to_list()
Y_train_list=y_train.to_list()
X_test_list=X_test.to_list()
Y_test_list=y_test.to_list()
train_data=[]
train_labels=[]
test_data=[]
test_labels=[]
for i in range(0,len(X_train_list)):
  if type(X_train_list[i])==str:
    train_data.append(X_train_list[i])
    train_labels.append(Y_train_list[i])
for i in range(0,len(X_test_list)):
  if type(X_test_list[i])==str:
    test_data.append(X_test_list[i])
    test_labels.append(Y_test_list[i])

df_train_tweet = pd.DataFrame (train_data,columns=['A'])
df_train_labels= pd.DataFrame (train_labels, columns=['B'])

print(type(df_train_tweet))
train_df=pd.DataFrame()
train_df['doc_len'] = df_train_tweet['A'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)

print(df_train_tweet['A'])
print(train_df['doc_len'])
print(max_seq_len)
print(len(train_labels))

#visualize word distribution
sns.distplot(train_df['doc_len'], hist=True, kde=True, color='b', label='doc len')
plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
plt.title('comment length'); plt.legend()
plt.show()

label_names=[0,1]
processed_docs_train = train_data
processed_docs_test = test_data
num_classes = len(label_names)

# print("pre-processing train data...")
# processed_docs_train = []
# for doc in tqdm(raw_docs_train):
#     print(doc)
#     tokens = tokenizer.tokenize(doc)
#     filtered = [word for word in tokens if word not in stop_words]
#     processed_docs_train.append(" ".join(filtered))
# #end for
# MAX_NB_WORDS = 100000
# processed_docs_test = []
# for doc in tqdm(raw_docs_test):
#     tokens = tokenizer.tokenize(doc)
#     filtered = [word for word in tokens if word not in stop_words]
#     processed_docs_test.append(" ".join(filtered))
# #end for
print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky

print((processed_docs_test))
print(len(processed_docs_train))

word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))
print(len(word_seq_train))
print(len(word_seq_test))
#pad sequences

word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
print(len(word_seq_train))
print(len(word_seq_test))



batch_size = 256
num_epochs = 8 

#model parameters
num_filters = 64 
embed_dim = 300 
weight_decay = 1e-4

"""** EMBEDDING MATRIX**"""

#embedding matrix
print('preparing embedding matrix...')
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#CNN architecture
print("training CNN ...")
model = Sequential()
model.add(Embedding(nb_words, embed_dim,
          weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

#define callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

print(len(word_seq_train))
print(len(train_labels))
data=pd.DataFrame(word_seq_train)
print(data)
labels_l=pd.DataFrame(train_labels)

test_data=pd.DataFrame(word_seq_test)
labels_tst=pd.DataFrame(test_labels)

hist = model.fit(data, labels_l, batch_size=64, epochs=num_epochs, callbacks=callbacks_list,  shuffle=True, verbose=2,validation_data=(test_data, labels_tst))

fpred=model.predict_classes(test_data)
print(fpred)

import pickle
pickle_out=open("prediction_files\\y_pred_fasttext_glove.p","wb")
pickle.dump(fpred,pickle_out)
pickle_out.close()

pickle_out=open("prediction_files\\Y_test_fasttext_glove_ACTUAL.p","wb")
pickle.dump(labels_tst,pickle_out)
pickle_out.close()

predictions=fpred
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

accuracy = accuracy_score(labels_tst, fpred)

print("Accuracy : %.2f%%" % (accuracy * 100.0))

print(precision_recall_fscore_support(labels_tst, predictions, average=None,labels=[0,1]))
print("Average Precision : ",precision_score(labels_tst, predictions, average='weighted'))
print("Recall weighted : ",recall_score(labels_tst, predictions, average='weighted'))
print("Roc_auc score : ",roc_auc_score(labels_tst, predictions,average='weighted'))

f1 = f1_score(labels_tst, predictions, average='macro')
print("FI Score" , f1)

print("Testing Classification report")
print(classification_report(labels_tst, predictions, target_names=['0','1']))
