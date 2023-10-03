#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:59:05 2023

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

chosen_X = 'Description_Processed'
X = df[chosen_X]
y = df['Class_Numeric']

#________________________________________TRAIN-TEST SPLIT________________________________________
from sklearn.model_selection import train_test_split

train_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42, shuffle=True)


#________________________________________TRAIN Word2Vec Model________________________________________
from gensim.models import Word2Vec

X_w2v = [text.split() for text in X]
w2v_model_64 = Word2Vec(X_w2v, sg=1, vector_size=64, window=5, min_count=4, workers=4)

# words closest to [website]  : 
# print(w2v_model_64.wv.most_similar("[website]"))
print(w2v_model_64.wv.most_similar("fraud"))
print(w2v_model_64.wv.most_similar("athletic"))

#________________________________________TOKENIZE ________________________________________
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df['Reduced_Processed_Length'] = [ len(str(text).split()) for text in df['Description_Reduced_Processed']]
df['Reduced_Length'] = [ len(str(text).split()) for text in df['Description_Reduced']]
df['Processed_Length'] = [ len(str(text).split()) for text in df['Description_Processed']]

print("99.9% records have less than ", df['Reduced_Processed_Length'].quantile(0.999), " tokens in Description_Reduced_Processed")
print("99.9% records have less than ", df['Reduced_Length'].quantile(0.999), " tokens in Description_Reduced")
print("99.9% records have less than ", df['Processed_Length'].quantile(0.999), " tokens in Processed_Length")

max_length = 70
padding_type = 'pre'

# returns words their tokens and frequencies
def get_token_frequencies(tokenizer):
    token_frequencies_df = pd.DataFrame.from_dict({ "word": list(tokenizer.word_counts.keys()), "token_frequency": list(tokenizer.word_counts.values()) } )
    token_frequencies_df['token'] = token_frequencies_df['word'].map(tokenizer.word_index)
    return token_frequencies_df

# computer vocabulary sized when removing tokens with a frequency < N 
def optimize_corpus_body(token_frequencies_df, corpus_size, min_token_frequency = 2):
    freq_distribution = (token_frequencies_df.token_frequency.value_counts().reset_index().head(15))
    unwanted_tokens_dist = freq_distribution[freq_distribution["index"] < min_token_frequency ]
    num_unwanted_tokens = sum(unwanted_tokens_dist["token_frequency"])
    print(" optimal_num_words : ", corpus_size - num_unwanted_tokens)
    return corpus_size - num_unwanted_tokens

# Tokenize the text data
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
tokenizer.fit_on_texts(X)
token_freq_df = get_token_frequencies(tokenizer)

# num_words = 82347

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen = max_length, padding=padding_type)  
X_test =  pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen = max_length, padding=padding_type)  
vocab_size = len(tokenizer.word_index) + 1

#________________________________________Create Word2Vec Embeddings________________________________________

def embedd_matrix(tokenizer, w2v_model, vocab_size ):
    # Create a weight matrix for the embedding layer
    embedding_matrix = np.zeros((vocab_size, w2v_model.vector_size))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    return embedding_matrix

embedding_matrix_64 = embedd_matrix(tokenizer, w2v_model_64, vocab_size)


#___________________________SAVE ANALYSIS HYPERTUNING PARAMS___________________________

cnn_lstm_performance = pd.read_csv("MODEL_PERFORMANCE/cnn_lstm_performance.csv")
# cnn_lstm_performance = pd.DataFrame({},
#             columns=[ 'X', 'labels_count', 'train_size', 
#                       'input_dim/vocab_size', 'output_dim/embedd_matrix_size', 'use_W2V', 'max_sequence_length', 
#                       'Conv1D_units', 'kernel_sizes', 'pool_sizes', 'LSTM_units','Dropout',          
#                       'learning_rate', 'batch_size', 'epochs', 'epochs_stopped',  
#                       'Val_loss', 'Val_accuracy', 'training_time_total'])
# cnn_lstm_performance.to_csv("MODEL_PERFORMANCE/cnn_lstm_performance.csv", index = False)


#___________________________NETWORK ARCHITECTURE___________________
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.optimizers import Adam
import time


def cnn_lstm_model(input_dim, output_dim, input_length, Conv1D_units, kernel_sizes, LSTM_units, Dropout_rate, uniq_labels_count, embedding_matrix=[]  ):
    model_RCNN = Sequential() 
    if(len(embedding_matrix)==0):
        model_RCNN.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length=input_length, trainable=True))
    else:
        model_RCNN.add(Embedding(input_dim = input_dim, output_dim = output_dim, weights=[embedding_matrix], input_length=input_length, trainable=False))
    model_RCNN.add(Conv1D(filters=Conv1D_units[0], kernel_size=kernel_sizes[0], padding='same', activation='relu')) 
    model_RCNN.add(MaxPooling1D(pool_size=2)) 
    model_RCNN.add(LSTM(LSTM_units[0])) 
    model_RCNN.add(Dropout(rate=Dropout_rate[0])) 
    model_RCNN.add(Dense(uniq_labels_count, activation='softmax'))
    model_RCNN.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    print(model_RCNN.summary())
    return model_RCNN

def train_cnn_lstm_model(model, X_train, y_train, epochs, batch_size, X_test,y_test, use_w2v, callbacks_list):
    start_training = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose = 1, shuffle=True, validation_data=(X_test,y_test), callbacks = callbacks_list)
    tot_time_trained = round(time.time() - start_training, 0)
    # get model metrics
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    training_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']
    epochs_trained = range(1, len(training_loss) + 1)

    # save analysis params
    cnn_lstm_performance.loc[len(cnn_lstm_performance)] = [
        chosen_X, uniq_labels_count,  train_size, 
        input_dim, output_dim, use_w2v, X_train.shape[1]   , 
        Conv1D_units, kernel_sizes, pool_sizes, LSTM_units, Dropout_rate,          
        lr, batch_size, epochs, epochs_trained,  
        min(test_loss), max(test_acc), tot_time_trained]

    cnn_lstm_performance.to_csv("MODEL_PERFORMANCE/cnn_lstm_performance.csv", index = False)
    
    return history


# embedding params 
input_dim = vocab_size
output_dim = 64
input_length = X_train.shape[1]
embedding_matrix = embedding_matrix_64

# layer params 
Conv1D_units = [48]
kernel_sizes = [4]
pool_sizes = [2]
LSTM_units = [32]
Dropout_rate = [0.15]
uniq_labels_count = len(np.unique(y_train))

# train params
lr = 0.008
batch_size = 256
epochs = 1
callbacks_list = [EarlyStopping(patience=2, start_from_epoch=1, monitor = 'val_accuracy' , mode='auto', min_delta= 0.01  )]


model_RCNN = cnn_lstm_model(input_dim, output_dim, input_length, Conv1D_units, kernel_sizes, LSTM_units, Dropout_rate, uniq_labels_count )
history = train_cnn_lstm_model(model_RCNN, X_train, y_train, epochs, batch_size, X_test, y_test, False, callbacks_list)


model_RCNN_w2v = cnn_lstm_model(input_dim, output_dim, input_length, Conv1D_units, kernel_sizes, LSTM_units, Dropout_rate, uniq_labels_count, embedding_matrix )
history_w2v = train_cnn_lstm_model(model_RCNN_w2v, X_train, y_train, epochs, batch_size, X_test, y_test, True, callbacks_list)


#________________________________________SAVE MODEL PERFORMANCE________________________________________

from keras.utils.vis_utils import plot_model
plot_model(model_RCNN,  show_shapes=False, show_dtype=False,show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)


import visualkeras
visualkeras.layered_view(model_RCNN, to_file='output_RCNN.png')

model_json = model_RCNN.to_json()
with open("SAVED_MODELS\\model_RCNN_.json", "w") as json_file:
    json_file.write(model_json)

#___________________________EVALUATING LOSS and ACCURACY___________________
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sns.set_style("white")
sns.color_palette("coolwarm", as_cmap=True)

training_loss = history.history['loss']
test_loss = history.history['val_loss']
training_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
epochs_trained = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epochs_trained, training_loss, 'r--')
plt.plot(epochs_trained, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LOSS vs EPOCHS')
plt.show();

# Visualize accuracy history
plt.plot(epochs_trained, training_acc, 'r--')
plt.plot(epochs_trained, test_acc, 'b-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ACCURACY vs EPOCHS')
plt.show();




#___________________________EVALUATING MODEL ACCURACY___________________

target_names_sorted = dict(sorted(labels_class_mapping.items())).values()

y_pred_test_raw = model_RCNN.predict(X_test)
y_pred_test_labels = np.argmax(y_pred_test_raw, axis=1)
test_report = classification_report(y_test, y_pred_test_labels, target_names = target_names_sorted   )

y_pred_train_raw = model_RCNN.predict(X_train)
y_pred_train_labels = np.argmax(y_pred_train_raw, axis=1)
full_data_report = classification_report(y_train, y_pred_train_labels)
                                     
print("TEST DATASET REPORT : \n", test_report)
print("FULL DATASET REPORT    : \n", full_data_report)

conf_mat_test = pd.DataFrame(confusion_matrix(y_test, y_pred_test_labels))  
conf_mat_test = conf_mat_test.rename(columns=labels_class_mapping, index=labels_class_mapping)

conf_mat_train = pd.DataFrame(confusion_matrix(y_train, y_pred_train_labels ))  
conf_mat_train = conf_mat_train.rename(columns=labels_class_mapping, index=labels_class_mapping)

fig = plt.figure(figsize=(10, 7))  
sns.heatmap(conf_mat_test, annot=True, annot_kws={"size": 16}, fmt="g", cmap="rocket_r")  
plt.title("TEST Confusion Matrix")  
plt.xlabel("Predicted Label")  
plt.ylabel("True Label")  
plt.show()  

fig = plt.figure(figsize=(10, 7))  
sns.heatmap(conf_mat_train, annot=True, annot_kws={"size": 16}, fmt="g", cmap="rocket_r")  
plt.title("TRAIN Confusion Matrix")  
plt.xlabel("Predicted Label")  
plt.ylabel("True Label")  
plt.show()



