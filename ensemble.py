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
import pickle
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D,Bidirectional
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from sklearn.metrics import roc_curve, auc,precision_recall_fscore_support, f1_score,roc_auc_score,accuracy_score, classification_report
from imblearn.metrics import sensitivity_specificity_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def print_metrics(X_train,y_train,y_test,predictions):
    accuracy = accuracy_score(y_test, predictions)
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


# In[12]:



data=pd.read_csv("Preprocessed_Final_Datatset_new.csv",encoding='latin1')


# In[13]:


print("before split")
count_vectorizer = CountVectorizer(stop_words='english') 
cv = count_vectorizer.fit_transform(data['Clean_TweetText'].values.astype('U'))
X_train,X_test,Y_train,Y_test = train_test_split(cv,data['label'] , test_size=.2,stratify=data['label'], random_state=42)
print("after split")
pickle_out1=open("y_test_ensemble_word2vec.p","wb")
pickle.dump(Y_test,pickle_out1)
pickle_out1.close()

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
clf1 = LogisticRegression(random_state=1)
clf2 = XGBClassifier()

eclf = VotingClassifier(estimators=[('lr', clf1), ('xgb', clf2)], voting='soft')

for clf, label in zip([clf1, clf2, eclf], ['Logistic Regression', 'XGBoost','Ensemble']):
    scores = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

eclf.fit(X_train, Y_train)
predict_ensemble=eclf.predict(X_test)
print_metrics(X_train,Y_train,Y_test,predict_ensemble)

pickle_out=open("y_pred_ensemble_word2vec.p","wb")
pickle.dump(predict_ensemble,pickle_out)
pickle_out.close()