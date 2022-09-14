# -*- coding: utf-8 -*-
"""simple_rnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dX9NKlnuVsFkpch9B7rdEfnxql7Q7JFK
"""

from google.colab import drive
drive.mount('/content/drive')

import nltk
import pandas as pd
import codecs
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import string

import re
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

nltk.download('stopwords')
nltk.download('punkt')



text=pd.read_csv('/content/drive/MyDrive/ML Assignment/Assignment_6/Preprocessed_Final_Datatset_new.csv',encoding='latin1')

text_list=text['Clean_TweetText'].to_list()

print(text)
print(text_list)

from collections import Counter
label_list=text['label'].to_list()
label_list=[1 if x==4 else x for x in label_list]
print(Counter(label_list))











all_tokensList=[]
whole_tokenList=[]
labels=[]
c=0
for each_word in text_list:
  if type(each_word)==str:
    tokens=word_tokenize(each_word) # word tokenization
    table=str.maketrans('','',string.punctuation) # remove punctuation
    ptokens=[w.translate(table) for w in tokens]
    #print(ptokens)
    normal_tokens=[p.lower() for p in ptokens] # convert in to lowercase
    stop_words = set(stopwords.words('english')) # stopwords removal
    words=[t1 for t1 in normal_tokens if not t1 in stop_words]
    non_blank_tokens = [s1 for s1 in words if s1] # remove blank tokens
    r="".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in non_blank_tokens]).strip()
    all_tokensList.append(r)
    labels.append(label_list[c])
    c+=1
print(all_tokensList)

#df['text']=all_tokensList
#df['target']=labels

df1=pd.DataFrame()
df1['text']=all_tokensList
df1['target']=labels
X=list(df1['text'])
Y=list(df1['target'])

df1

import seaborn as sns
sns.set_theme(style="darkgrid")
sns.countplot(x='target', data=df1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
print(X_train)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print(X_train)

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

maxlen = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train)
print(X_test)

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('/content/drive/MyDrive/ML Assignment/Assignment_6/glove.6B.50d.txt', encoding="utf8")

for line in glove_file:
    #print(line)
    records = line.split()
    word = records[0]
    #print(word)
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
    #print(embeddings_dictionary)
glove_file.close()

embedding_matrix = zeros((vocab_size, 50))
vector_list=[]
each_token_list=[]
for word, index in tokenizer.word_index.items():
    #print(word,index)
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        vector_list.append(embedding_vector)
        each_token_list.append(word)

def tsne():
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(vector_list[:100])
    print(new_values.shape)
    
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16,16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(each_token_list[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

from keras.layers import SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
model = Sequential()
embedding_layer = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(SimpleRNN(64))
#model.add(Dense(32,activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(64,activation='tanh'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))



import keras
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Create callbacks
callbacks = [ModelCheckpoint('/content/drive/MyDrive/ML Assignment/Assignment_6/model1reluhrms.h5',save_best_only=True, monitor='val_loss')]

import numpy as np
history = model.fit(np.array(X_train), np.array(y_train), batch_size=256, epochs=6, verbose=1,callbacks=callbacks,
                    validation_data=(np.array(X_test), np.array(y_test)))
model.save("Q1_model")



model=keras.models.load_model('/content/drive/MyDrive/ML Assignment/Assignment_6/model1reluhrms.h5')
score = model.evaluate(np.array(X_test), np.array(y_test), verbose=1)



score_train = model.evaluate(np.array(X_train), np.array(y_train), verbose=1)

predictions=list(model.predict_classes(X_test))

print(predictions)

print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

print("Train Loss:", score_train[0])
print("Train Accuracy:", score_train[1])

fpred=predictions
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test, fpred)

print("Accuracy : %.2f%%" % (accuracy * 100.0))

print(precision_recall_fscore_support(y_test, predictions, average=None,labels=[0,1]))
print("Average Precision : ",precision_score(y_test, predictions, average='weighted'))
print("Recall weighted : ",recall_score(y_test, predictions, average='weighted'))
print("Roc_auc score : ",roc_auc_score(y_test, predictions,average='weighted'))

f1 = f1_score(y_test, predictions, average='macro')
print("FI Score" , f1)

print("Testing Classification report")
print(classification_report(y_test, predictions, target_names=['0','1']))


print("\n")

import pickle
pickle_out=open("prediction_files\\y_pred_simple_rnn_glove.p","wb")
pickle.dump(fpred,pickle_out)
pickle_out.close()

pickle_out=open("prediction_files\\Y_test_simple_rnn_glove_ACTUAL.p","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

tsne()

