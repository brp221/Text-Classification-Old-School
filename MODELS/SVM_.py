#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:52:42 2023

@author: bratislavpetkovic
"""
import numpy as np 
import pandas as pd
import os 
import seaborn as sns
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

path = "/Users/bratislavpetkovic/Desktop/MediumArticles/TextClassification/"
os.chdir(path)

#________________________________________GET DATA________________________________________

IMDB_data = "IMDB_Dataset/imdb_df_preprocessed.csv"
AG_NEWS_data = "AG_News_Dataset/ag_news_df_preprocessed.csv"

df = pd.read_csv(AG_NEWS_data)
df['Class_Numeric'] = df['Class'].astype('category').cat.codes
df = df.dropna()

class_labels_mapping = dict(zip(df.Class, df.Class_Numeric)) 
labels_class_mapping = {v: k for k, v in class_labels_mapping.items()}

print(df.Class_Numeric.value_counts())

chosen_X = 'Description_Reduced_Processed'
X = df[chosen_X]
y = df['Class_Numeric']


#________________________________________VECTORIZE________________________________________

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=10000)
BOW = vectorizer.fit_transform(X)

#________________________________________TRAIN-TEST SPLIT________________________________________
from sklearn.model_selection import train_test_split

train_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(BOW, y, test_size=1-train_size, random_state=42, shuffle=True)



#________________________________________TRAIN MODEL________________________________________
from sklearn.svm import SVC
import time 

svm_model = SVC( probability=True)

start = time.time()
svm_model.fit(X_train,y_train, verbose = True)
end = time.time()
total = end - start 

#________________________________________EVALUATE PERFORMANCE________________________________________

from sklearn.metrics import  classification_report


y_test_pred_prob = svm_model.predict_proba(X_test)
y_pred_test = np.argmax(y_test_pred_prob, axis=1)
test_report = classification_report(y_test, y_pred_test   )


print(test_report)

#________________________________________SAVE MODEL________________________________________

import pickle
from joblib import dump, load

dump(svm_model, 'SAVED_MODELS/SVM_model.joblib')
svm_model = load('SAVED_MODELS/SVM_model.joblib')













