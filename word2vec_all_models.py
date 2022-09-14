#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc,precision_recall_fscore_support, f1_score,roc_auc_score,accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold , f_classif
from sklearn.model_selection import train_test_split , cross_val_predict,cross_val_score,ShuffleSplit
from sklearn import model_selection
from sklearn import svm
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding


# In[2]:



data=pd.read_csv("Preprocessed_Final_Datatset_new.csv",encoding='latin1')

print("before split")

X_train,X_test,y_train,y_test = train_test_split(data['Clean_TweetText'],data['label'] , test_size=.2,stratify=data['label'], random_state=42)

print("after split")


# In[6]:


# integer encode the documents
vocab_size = 50
labels = data['label']
docs = data['Clean_TweetText'].to_frame()
print(labels[5])
print(docs)
count=-1
encoded_docs=[]
new_labels=[]
for index,d in docs.iterrows():
    text = d['Clean_TweetText']
    if type(text) == str:
        encoded_docs.append(one_hot(text, vocab_size))
        new_labels.append(labels[index])



print(len(encoded_docs))
print(len(new_labels))

# pad documents to a max length of 4 words
max_length = 10
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(padded_docs, new_labels, test_size=0.3, random_state=42 ,stratify=new_labels,shuffle=True)


# In[9]:


# define the model
import numpy as np 
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))
# compile the model
print("here")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
print("Now")
model.fit(X_train, Y_train, epochs=3, verbose=1, validation_data=(X_test, Y_test))
print("Done")
test_loss, test_acc = model.evaluate(X_test,Y_test,verbose=0)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[10]:


test_loss, test_acc = model.evaluate(X_test,Y_test,verbose=0)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


# In[17]:



predictions = model.predict_classes(X_test) 
#Undo scaling
#predictions = scaler.inverse_transform(predictions)
import pickle

fpred=[]
for i in predictions:
    j=list(i)
    fpred.append(j[0])

pickle_out=open("prediction_files\\y_pred_seq_word2vec.p","wb")
pickle.dump(fpred,pickle_out)
pickle_out.close()

pickle_out=open("prediction_files\\Y_test_seq_word2vecACTUAL.p","wb")
pickle.dump(Y_test,pickle_out)
pickle_out.close()
print(len(predictions))
print(len(Y_test))


predictions = fpred
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(Y_test, fpred)
print("Accuracy : %.2f%%" % (accuracy * 100.0))

print("Average Precision : ",precision_score(Y_test, predictions, average='weighted'))
print("Recall weighted : ",recall_score(Y_test, predictions, average='weighted'))
print("Roc_auc score : ",roc_auc_score(Y_test, predictions,average='weighted'))

f1 = f1_score(Y_test, predictions, average='macro')
print("FI Score" , f1)

print("Testing Classification report")
print(classification_report(Y_test, predictions, target_names=['0','1']))


print("\n")


# In[21]:


from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers import Dense, Flatten, Conv2D, MaxPooling1D,ZeroPadding2D,BatchNormalization
from keras.layers import Dense, LSTM
from keras.layers import Dense, LSTM
from keras.optimizers import Adam,RMSprop

model = Sequential()
model.add(Embedding(vocab_size, 32))

model.add(Conv1D(32, 5, activation='relu'))
model.add(MaxPooling1D(5))

model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.5))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.001), loss = 'binary_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(X_train, Y_train, batch_size=32, epochs=2, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

predictions = model.predict_classes(X_test) 
#Undo scaling
#predictions = scaler.inverse_transform(predictions)
import pickle

fpred=[]
for i in predictions:
    j=list(i)
    fpred.append(j[0])

pickle_out=open("prediction_files\\y_pred_cnn_word2vec.p","wb")
pickle.dump(fpred,pickle_out)
pickle_out.close()

pickle_out=open("prediction_files\\Y_test_cnn_word2vecACTUAL.p","wb")
pickle.dump(Y_test,pickle_out)
pickle_out.close()
print(len(predictions))
print(len(Y_test))


predictions = fpred
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(Y_test, fpred)
print("Accuracy : %.2f%%" % (accuracy * 100.0))

print("Average Precision : ",precision_score(Y_test, predictions, average='weighted'))
print("Recall weighted : ",recall_score(Y_test, predictions, average='weighted'))
print("Roc_auc score : ",roc_auc_score(Y_test, predictions,average='weighted'))

f1 = f1_score(Y_test, predictions, average='macro')
print("FI Score" , f1)

print("Testing Classification report")
print(classification_report(Y_test, predictions, target_names=['0','1']))


print("\n")


# In[22]:


import pickle


pickle_out=open("prediction_files\\y_pred_cnn_word2vec.p","wb")
pickle.dump(predictions,pickle_out)
pickle_out.close()

pickle_out=open("prediction_files\\Y_test_cnn_word2vecACTUAL.p","wb")
pickle.dump(Y_test,pickle_out)
pickle_out.close()


# In[ ]:




