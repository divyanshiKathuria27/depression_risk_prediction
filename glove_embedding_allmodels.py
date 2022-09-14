#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc,precision_recall_fscore_support, f1_score,roc_auc_score,accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold , f_classif
from sklearn.model_selection import train_test_split , cross_val_predict,cross_val_score,ShuffleSplit
from sklearn import model_selection
from sklearn import svm
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

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


# In[63]:



data=pd.read_csv("Preprocessed_Final_Datatset_new.csv",encoding='latin1')


labels = data['label']
data = data.drop(['label'],axis = 1)
print("before split")

X_train_full,X_test_full,Y_train,Y_test = train_test_split(data,labels, test_size=.2,stratify=labels, random_state=42)

print("after split")



# In[86]:



df_train = pd.DataFrame(data=X_train_full)
df_test = pd.DataFrame(data=X_test_full)

X_train = X_train_full['Clean_TweetText'].astype('U')
X_test = X_test_full['Clean_TweetText'].astype('U')

date_test = X_test_full['date']
date_train = X_train_full['date']


print(X_test)
print(date_test)


# In[87]:


import datetime
month_list=[]
for row in date_test:
    
    serial = row
    seconds = (serial - 25569) * 86400.0
    obj = datetime.datetime.utcfromtimestamp(seconds)
    month_list.append(obj.month)


# In[88]:


print(len(month_list))
print(month_list.count(1))
print(month_list.count(2))
print(month_list.count(3))
print(month_list.count(4))
print(month_list.count(5))
print(month_list.count(6))
print(month_list.count(7))
print(month_list.count(8))


# In[89]:



tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[90]:


print(X_train)


# In[91]:


# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[92]:


from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


# In[93]:


embedding_matrix = zeros((vocab_size, 100))
count=0
c=0
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        c=c+1
    else:
        count=count+1


# In[94]:


count


# In[95]:


c


# In[96]:


model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())
history = model.fit(X_train, Y_train, batch_size=128, epochs=30, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, Y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[97]:


predictions = model.predict_classes(X_test) 
#Undo scaling
#predictions = scaler.inverse_transform(predictions)
print(predictions)

fpred=[]
for i in predictions:
    j=list(i)
    fpred.append(j[0])
print(fpred)

pickle_out=open("prediction_files\\y_pred_seq_glove.p","wb")
pickle.dump(fpred,pickle_out)
pickle_out.close()

pickle_out=open("prediction_files\\Y_test_seq_glove_ACTUAL.p","wb")
pickle.dump(Y_test,pickle_out)
pickle_out.close()


# In[98]:


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


# In[99]:


predictions = fpred
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(Y_test, fpred)
print("Accuracy : %.2f%%" % (accuracy * 100.0))

print(precision_recall_fscore_support(Y_test, predictions, average=None,labels=[0,1]))
print("Average Precision : ",precision_score(Y_test, predictions, average='weighted'))
print("Recall weighted : ",recall_score(Y_test, predictions, average='weighted'))
print("Roc_auc score : ",roc_auc_score(Y_test, predictions,average='weighted'))

f1 = f1_score(Y_test, predictions, average='macro')
print("FI Score" , f1)

print("Testing Classification report")
print(classification_report(Y_test, predictions, target_names=['0','1']))


print("\n")


# In[112]:


dict_months={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

for i in range(0,len(month_list)):
    if predictions[i] == 1:
        dict_months[month_list[i]] = dict_months[month_list[i]]+1
print(dict_months)

for key in dict_months.keys():
    dict_months[key] = dict_months[key]/(month_list.count(key))

print(dict_months)


# In[131]:


dict_new = {'Jan':0,'Feb':0,'March':0,'April':0,'May':0,'June':0,'July':0,'Aug':0,'September':0}
count=0
for key in dict_new.keys():
    count=count+1
    dict_new[key] = dict_months[count]

print(dict_new)
print(dict_months)
import matplotlib.pyplot as plt
plt.plot(range(len(dict_new)), list(dict_new.values()),marker='o')
plt.xticks(range(len(dict_new)), list(dict_new.keys()))
plt.xlabel('Months')
plt.ylabel('Rate of depression')
plt.grid(True, linewidth=0.5, color='grey', linestyle='-')
plt.show()


# In[118]:


from keras.layers import SimpleRNN
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,ZeroPadding2D,BatchNormalization
from keras.optimizers import Adam,RMSprop
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(SimpleRNN(64))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))

#model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

import keras
add =  keras.optimizers.RMSprop(lr=0.001)
model.compile(optimizer=add, loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, Y_train, batch_size=32, epochs=6, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, Y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[ ]:


predictions = model.predict_classes(X_test) 
#Undo scaling
#predictions = scaler.inverse_transform(predictions)
print(predictions)

fpred=[]
for i in predictions:
    j=list(i)
    fpred.append(j[0])
print(fpred)

pickle_out=open("prediction_files\\y_pred_simplernn_glove.p","wb")
pickle.dump(fpred,pickle_out)
pickle_out.close()

pickle_out=open("prediction_files\\Y_test_simplernn_glove_ACTUAL.p","wb")
pickle.dump(Y_test,pickle_out)
pickle_out.close()


# In[123]:


from keras.layers import Dense, LSTM
from keras.optimizers import Adam,RMSprop

x_train, y_train = np.array(X_train), np.array(Y_train)
x_test,y_test = np.array(X_test), np.array(Y_test)


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print("here")

#### Model 1: LSTM Loss function = binary_crossentropy ####
model = Sequential()

model.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 100))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer=Adam(learning_rate=0.01), loss = 'binary_crossentropy')
model.fit(x_train, y_train, epochs = 3, batch_size = 128,verbose=1)

predictions = model.predict_classes(x_test) 
#Undo scaling
#predictions = scaler.inverse_transform(predictions)


fpred=[]
for i in predictions:
    j=list(i)
    fpred.append(j[0])


pickle_out=open("prediction_files\\y_pred_lstm_glove.p","wb")
pickle.dump(fpred,pickle_out)
pickle_out.close()

pickle_out=open("prediction_files\\Y_test_lstm_glove_ACTUAL.p","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()
import pickle
predictions = fpred
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(Y_test, fpred)
print("Accuracy : %.2f%%" % (accuracy * 100.0))

print(precision_recall_fscore_support(Y_test, predictions, average=None,labels=[0,1]))
print("Average Precision : ",precision_score(Y_test, predictions, average='weighted'))
print("Recall weighted : ",recall_score(Y_test, predictions, average='weighted'))
print("Roc_auc score : ",roc_auc_score(Y_test, predictions,average='weighted'))

f1 = f1_score(Y_test, predictions, average='macro')
print("FI Score" , f1)

print("Testing Classification report")
print(classification_report(Y_test, predictions, target_names=[0,1]))


print("\n")


# In[ ]:





# In[ ]:


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
history = model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[ ]:


print(score)


# In[ ]:



predictions = model.predict_classes(X_test) 
#Undo scaling
#predictions = scaler.inverse_transform(predictions)
import pickle
fpred=[]
for i in predictions:
    j=list(i)
    fpred.append(j[0])

import pickle
pickle_out=open("prediction_files\\y_pred_cnn_glove.p","wb")
pickle.dump(fpred,pickle_out)
pickle_out.close()

pickle_out=open("prediction_files\\Y_test_cnn_glove_ACTUAL.p","wb")
pickle.dump(Y_test,pickle_out)
pickle_out.close()


# In[43]:


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


# In[44]:


print(accuracy_score(Y_test,fpred))


# In[46]:


predictions = fpred
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(Y_test, fpred)
print("Accuracy : %.2f%%" % (accuracy * 100.0))

print(precision_recall_fscore_support(Y_test, predictions, average=None,labels=[0,1]))
print("Average Precision : ",precision_score(Y_test, predictions, average='weighted'))
print("Recall weighted : ",recall_score(Y_test, predictions, average='weighted'))
print("Roc_auc score : ",roc_auc_score(Y_test, predictions,average='weighted'))

f1 = f1_score(Y_test, predictions, average='macro')
print("FI Score" , f1)

print("Testing Classification report")
print(classification_report(Y_test, predictions, target_names=['0','1']))


print("\n")


# In[ ]:




