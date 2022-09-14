import pandas as pd
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc,precision_recall_fscore_support, f1_score,roc_auc_score,accuracy_score, classification_report
from sklearn.model_selection import train_test_split , cross_val_predict,cross_val_score,ShuffleSplit
import numpy as np
import nltk
import pickle
from nltk.corpus import stopwords
from numpy import array
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

data=pd.read_csv("Preprocessed_Final_Datatset_new.csv",encoding='latin1')

print("before split")
count_vectorizer = CountVectorizer(stop_words='english') 
cv = count_vectorizer.fit_transform(data['Clean_TweetText'].values.astype('U'))
X_train,X_test,Y_train,Y_test = train_test_split(cv,data['label'] , test_size=.2,stratify=data['label'], random_state=42)
print("after split")
'''
pickle_out=open("y_test_logreg_word2vec.p","wb")
pickle.dump(Y_test,pickle_out)
pickle_out.close()
'''
clf = LogisticRegression(random_state=1)
clf.fit(X_train, Y_train)
predict_logreg=clf.predict(X_test)
print_metrics(X_train,Y_train,Y_test,predict_logreg)
'''
pickle_out1=open("y_pred_logreg_word2vec.p","wb")
pickle.dump(predict_logreg,pickle_out1)
pickle_out1.close()
'''