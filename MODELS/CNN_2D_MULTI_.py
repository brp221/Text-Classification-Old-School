# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:36:56 2023

@author: U1255683
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
print(w2v_model_64.wv.most_similar("[website]"))
print(w2v_model_64.wv.most_similar("fraud"))
print(w2v_model_64.wv.most_similar("scary"))

#________________________________________TOKENIZE ________________________________________
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df['Reduced_Processed_Length'] = [ len(str(text).split()) for text in df['Description_Reduced_Processed']]
df['Reduced_Length'] = [ len(str(text).split()) for text in df['Description_Reduced']]

print("99.9% records have less than ", df['Reduced_Processed_Length'].quantile(0.999), " tokens in Description_Reduced_Processed")
print("99.9% records have less than ", df['Reduced_Length'].quantile(0.999), " tokens in Description_Reduced")

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

cnn_2d_performance = pd.read_csv("MODELS_PERFORMANCE\\cnn_lstm_performance.csv")
cnn_2d_performance = pd.DataFrame({},
            columns=[ 'X', 'labels_count', 'train_size', 
                      'input_dim/vocab_size', 'output_dim/embedd_matrix_size', 'use_W2V', 'max_sequence_length', 
                      'Conv1D_units', 'kernel_sizes', 'pool_sizes', 'LSTM_units','Dropout',          
                      'learning_rate', 'batch_size', 'epochs', 'epochs_stopped',  
                      'Val_loss', 'Val_accuracy', 'training_time_total'])
cnn_2d_performance.to_csv("MODELS_PERFORMANCE\\cnn_2d_performance.csv", index = False)

#___________________________SAVE ANALYSIS HYPERTUNING PARAMS___________________________


# cnn_rnn_hyperparameter_tuning_df = pd.read_csv("MODELS_PERFORMANCE\\cnn_rnn_hyperparameter_tuning_df.csv")
# # cnn_rnn_hyperparameter_tuning_df = pd.DataFrame({},
# #             columns=[ 'dataset_version', 'labels_count', 'train_size', 
# #                       'input_dim/vocab_size', 'output_dim/embedd_matrix_size', 'use_W2V', 'max_sequence_length', 
# #                       'Conv1D_units', 'kernel_sizes', 'pool_sizes', 'LSTM_units','Dropout',          
# #                       'learning_rate', 'batch_size', 'epochs', 'epochs_stopped',  
# #                       'Val_loss', 'Val_accuracy', 'training_time_total'])
# cnn_rnn_hyperparameter_tuning_df.to_csv("MODELS_PERFORMANCE\\cnn_rnn_hyperparameter_tuning_df.csv", index = False)
# cnn_1D_hyperparameter_tuning_df = pd.read_csv("models\\cnn_1D_hyperparameter_tuning_df.csv")




#___________________________NETWORK ARCHITECTURE___________________
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Conv2D, MaxPooling2D , Input, SpatialDropout1D, Reshape, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import time
from datetime import datetime

# embedding params 
input_dim = vocab_size
output_dim = 64
input_length = X_train.shape[1]
embedding_matrix = embedding_matrix_64
spatial_dropout = 0.2

# layer params 
Conv2D_filters = [ 64, 32, 32, 16]
kernel_sizes = [2,3,4,5, 6]
pool_sizes = [2, 2, 2, 2]
Dense_units = 64
uniq_labels_count = len(np.unique(y_train))

#train params
lr = 0.0075
batch_size = 300
epochs = 2
model_cp_path = "MODEL_CHECKPOINTS\\CNN_2D_MultiChannel\\model_weights__" + datetime.today().strftime('%Y-%m-%d') + "__.ckpt"

callbacks_list = [EarlyStopping(patience=2, start_from_epoch=3, monitor = 'val_accuracy' , mode='auto', min_delta= 0.01  ), 
                  ModelCheckpoint(filepath= model_cp_path, save_weights_only=True, verbose=1)]


def cnn_2d_multi():    
    inp = Input(shape=(max_length, ))
    x = Embedding(input_dim, output_dim, input_length = input_length, weights=[embedding_matrix])(inp) # trainable = True??? or False
    x = SpatialDropout1D(spatial_dropout)(x)
    x = Reshape((max_length, output_dim, 1))(x)
    
    conv_0 = Conv2D(Conv2D_filters[0], kernel_size=(kernel_sizes[0], output_dim), kernel_initializer='normal',activation='elu')(x)
    conv_1 = Conv2D(Conv2D_filters[0], kernel_size=(kernel_sizes[1], output_dim), kernel_initializer='normal',activation='elu')(x)
    conv_2 = Conv2D(Conv2D_filters[0], kernel_size=(kernel_sizes[2], output_dim), kernel_initializer='normal',activation='elu')(x)
    conv_3 = Conv2D(Conv2D_filters[0], kernel_size=(kernel_sizes[3], output_dim), kernel_initializer='normal',activation='elu')(x)
    conv_4 = Conv2D(Conv2D_filters[0], kernel_size=(kernel_sizes[4], output_dim), kernel_initializer='normal',activation='elu')(x)

    maxpool_0 = MaxPooling2D(pool_size=(max_length - kernel_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(max_length - kernel_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(max_length - kernel_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPooling2D(pool_size=(max_length - kernel_sizes[3] + 1, 1))(conv_3)
    maxpool_4 = MaxPooling2D(pool_size=(max_length - kernel_sizes[4] + 1, 1))(conv_3)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
        
    outp = Dense(uniq_labels_count, activation="softmax")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    print(model.summary())
    return model

def train_CNN_2D_model(model, X_train, y_train, epochs, batch_size, X_test,y_test, use_w2v, callbacks_list):
    start_training = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose = 1, 
                            shuffle=True, validation_data=(X_test , y_test), callbacks = callbacks_list) 
    tot_time_trained = round(time.time() - start_training, 0)
    
    #________________________________________SAVE MODEL PERFORMANCE________________________________________
    
    # model metrics
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    training_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']
    epochs_trained = range(1, len(training_loss) + 1)
    
    return history


cnn_2d_model = cnn_2d_multi()
history = train_CNN_2D_model(cnn_2d_model, X_train, y_train, 1, batch_size, X_test,y_test, True, callbacks_list)


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

y_pred_test_raw = cnn_2d_model.predict(X_test)
y_pred_test_labels = np.argmax(y_pred_test_raw, axis=1)
test_report = classification_report(y_test, y_pred_test_labels, target_names = target_names_sorted   )

y_pred_train_raw = cnn_2d_model.predict(X_train)
y_pred_train_labels = np.argmax(y_pred_train_raw, axis=1)
full_data_report = classification_report(y_train, y_pred_train_labels, target_names = target_names_sorted )
                                     
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







