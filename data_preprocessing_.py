#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:58:32 2023

@author: bratislavpetkovic
"""

import pandas as pd 
import os 
import seaborn as sns 
import numpy as np 

path = "/Users/bratislavpetkovic/Desktop/MediumArticles/TextClassification/"
os.chdir(path)


#__________________________________________READ __________________________________________

# AG NEWS : https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
AG_news_df = pd.read_csv("AG_News_Dataset/train.csv")
AG_news_df['Class'] = AG_news_df['Class Index']
AG_news_df = AG_news_df.replace({"Class" : {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}})
AG_news_df = AG_news_df.rename(columns = {'Class Index' : 'Class_Numeric'})


#__________________________________________EXPLORE __________________________________________
print(AG_news_df.Class.value_counts())

AG_news_df['Original_Length'] = [ len(str(text).split()) for text in AG_news_df['Description']]
sns.histplot(data=AG_news_df, x='Original_Length')

print(AG_news_df['Original_Length'].quantile(0.999))


# GPT costs :
# Training: $0.008 / 1K Tokens
# Usage input: $0.012 / 1K Tokens
# Usage output: $0.016 / 1K Tokens
# https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates

def cost_training(n_rows, total_token_count):
    # cost = rate * number of total tokens 
    training_cost = (0.008/1000) * total_token_count * 0.8 # assuming 80-20 training split 
    usage_input = (0.012/1000) * total_token_count * 0.2 
    usage_output = (0.016/1000) * n_rows * 0.2   # 1 token for each output 
    print("training_cost : " , training_cost)
    print("usage cost : " , usage_input + usage_output)
    print("total cost : " , training_cost + usage_input + usage_output)
    return training_cost + usage_input + usage_output

AG_news_GPT_cost = cost_training(len(AG_news_df), AG_news_df['Original_Length'].sum())
different_data_GPT_cost = cost_training(n_rows = 200000, total_token_count = 200000 * 600)

#__________________________________________SAMPLING __________________________________________

# balance dataset ? by injecting under and over sampling 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def undersample_dataset(df, sampling_strategy = "auto"):
    X, y = np.array(df['Description']), np.array(df['Class'])
    rus = RandomUnderSampler(random_state=0, sampling_strategy = sampling_strategy  )
    rus.fit(X.reshape(-1, 1), y)
    X_resampled, y_resampled = rus.fit_resample(X.reshape(-1, 1), y)
    print(X_resampled.shape)
    print(y_resampled.shape)

    undersampled_df = pd.DataFrame({"Description": X_resampled.flatten(), "Class":y_resampled.flatten()})
    
    print(undersampled_df.Class.value_counts())    
    return undersampled_df

def oversample_dataset(df, sampling_strategy = "auto"):
    X, y = np.array(df['Description']), np.array(df['Class'])
    ros = RandomOverSampler(random_state=0, sampling_strategy = sampling_strategy  )
    ros.fit(X.reshape(-1, 1), y)
    X_resampled, y_resampled = ros.fit_resample(X.reshape(-1, 1), y)
    print(X_resampled.shape)
    print(y_resampled.shape)

    oversampled_df = pd.DataFrame({"Description":X_resampled.flatten(), "Class":y_resampled.flatten()})
    
    print(oversampled_df.Class.value_counts())    
    return oversampled_df


#__________________________________________DATA AUGMENTATION__________________________________________
import spacy 
import re 
from tqdm import tqdm

# https://github.com/explosion/spacy-models/releases
# pip install /Users/bratislavpetkovic/Downloads/zh_core_web_lg-3.7.0.tar.gz 
# en_md_path =  path +"\\SPACY\\en_core_web_md-3.6.0\\en_core_web_md\\en_core_web_md-3.6.0" 
# spacy_nlp = spacy.load(en_md_path)

spacy_nlp = spacy.load('en_core_web_lg')

n = 2
regex_triple_letter = re.compile(r'(\w)(\1{2,})', flags=re.IGNORECASE) # set outside func as to not recompile every time 

def spacy_tagger(text, include_only = ["PERSON", "LAW", "CARDINAL", "MONEY", "PERCENT", "DATE"]):
    doc = spacy_nlp(text)
    for ent in doc.ents:
        if(ent.label_ in include_only):
            text = text.replace(ent.text, "[" + ent.label_ +"]")
    return text

def augment_text_spacy(text, remove_dates_ = True):
    # 1. remove anything between <> 
    text = re.sub( r'<[^>]+>', ' ', text)
    
    # 2. # apply minimal spell correction by removing triple letters
    text = regex_triple_letter.sub(r"\1", text)  
    
    # 3. identify websites
    new_pattern = r"(https://)?([a-z\.]+\.(com|org|co|us|uk|net|gov|edu)(/?))"
    text = re.sub(new_pattern, " [website] ", text) 
    text = re.sub(r'[a-z]*\[website\][a-z]*', " [website] ", text) # remove anything touching website 
    
    # 4. NER Tag via SpaCy (model = 'large' ? )
    text = spacy_tagger(text)
    
    # 5 Identify companies ? 
    text = re.sub(r'\s(inc\.|llc\.|llc|llp\.|llp|inc|corp\.|corp|ltd\.|ltd)\s',  " [company] " , text, flags=re.IGNORECASE) # insert company tags
    text = re.sub(r'[a-z]*\[company\][a-z]*', " [company] ", text).lower() # remove anything touching company 
    # text = re.sub( r'\s+\[company\](?:\s+\[company\])+', ' [company] ', text) # remove repeating [company] tags    
    
    # 6. Remove Dates 
    if(remove_dates_):
        text = re.sub(r'(\[DATE\])', ' ' , text, flags=re.IGNORECASE) # remove dates 
    
    # 7. TIDY WHITESPACES
    return re.sub(' +', ' ',text.lower()) # tidy whitespaces


def augment_data(df, remove_dates_ = True):
    df['Description_Reduced'] = [augment_text_spacy(text, remove_dates_) for text in tqdm([text for text in df['Description']])]
    return df 


#__________________________________________DATA PRE-PROCESSING__________________________________________

from spacy.symbols import ORTH

# add lemmatization exceptions
infixes = spacy_nlp.Defaults.infixes + [r"([\[\]])"]
spacy_nlp.tokenizer.infix_finditer = spacy.util.compile_infix_regex(infixes).finditer

# Add the special cases to the tokenizer
tags = ['cardinal','date','event','fac','gpe','language','law','loc','money','norp','ordinal','org','percent',
        'person','product','quantity','time','work_of_art', 'website', 'company', 'eos']
  
for tag in tags:
    spacy_nlp.tokenizer.add_special_case(f"[{tag}]", [{ORTH: f"[{tag}]"}])

def spacy_lemmatize(text):
    doc = spacy_nlp(text)
    return [token.lemma_ for token in doc]  #" ".join([token.lemma_ for token in doc])

keep_list = [ 'not', 'no', 'after', 'before', 'against']
stopwords_spaCy =  spacy_nlp.Defaults.stop_words
stopwords_custom =  [e for e in stopwords_spaCy if e not in keep_list]

def preprocess_NN(text, remove_stopwords = True):
    #1 Remove numbers and posessions 
    text = re.sub(r'\d+', ' ', text) 
    text = re.sub(r"('s|s')\b", ' ', text) #remove possesssions 
    
    #2 strip remaining punctuation 
    text = re.sub(r'[^\w\s\[\]]', " ", text)   # keep periods ? 

    #3 stopwords removal 
    if(remove_stopwords):
        text = [word for word in text.split() if word not in stopwords_custom]  # apply stopwords removal 
    
    #4 lemmatization
    text = spacy_lemmatize(" ".join(text))   
    
    #5 remove extra spaces
    return re.sub(' +', ' ',  " ".join(text) ) 



#__________________________________________BRING TOGETHER__________________________________________

############## IMDB DATASET ##############

# >>> 1st SAMPLE
upsample_strategy = {
    "War" : 10000, 
    "History" : 10000, 
    "Biography" : 8000, 
    "Animation" : 8000,
    "Sports" : 5000,
    "Film-noir" : 1500
    }
upsample_classes = list(upsample_strategy.keys()) 
upsample_df = oversample_dataset(imdb_df.query('Class in @upsample_classes'), upsample_strategy)

downsample_strategy = {
    "Thriller"  : 35000,
    "Romance"   : 35000,
    "Action"    : 35000,
    "Horror"    :   30740,
    "Crime"     :   29338,
    "Adventure" :   20422,
    "Mystery"   :   16578,
    "Scifi"     :   14412,
    "Fantasy"   :   14193,
    "Family"    :   13775
}
downsample_classes = list(downsample_strategy.keys())
downsample_df = undersample_dataset(imdb_df.query('Class in @downsample_classes'), downsample_strategy)

imdb_df_sampled = pd.concat([upsample_df, downsample_df])



# >>> 2nd Augment Data
imdb_df_sampled_augmented = augment_data(imdb_df_sampled)
imdb_df_sampled_augmented.to_csv("IMDB_Dataset/imdb_df_sampled_augmented.csv", index = False)


# >>> 3rd Pre Process Data
imdb_df_sampled_augmented['Description_Reduced_Processed'] = [preprocess_NN(text, remove_stopwords = True) for text in tqdm([text for text in imdb_df_sampled_augmented['Description_Reduced'] ])]
imdb_df_sampled_augmented.to_csv("IMDB_Dataset/imdb_df_preprocessed.csv", index = False)



############## AG NEWS DATASET ##############

# >>> 1st SAMPLE - no need already sampled
# >>> 2nd Augment Data
AG_news_df_sampled_augmented = augment_data(AG_news_df)
AG_news_df_sampled_augmented.to_csv("AG_News_Dataset/AG_news_df_sampled_augmented.csv", index = False)


# >>> 3rd Pre Process Data
AG_news_df_sampled_augmented['Description_Reduced_Processed'] = [preprocess_NN(text, remove_stopwords = True) for text in tqdm([text for text in AG_news_df_sampled_augmented['Description_Reduced'] ])]
AG_news_df_sampled_augmented['Description_Processed'] = [preprocess_NN(text, remove_stopwords = True) for text in tqdm([text for text in AG_news_df_sampled_augmented['Description'] ])]

AG_news_df_sampled_augmented.to_csv("AG_News_Dataset/AG_news_df_preprocessed.csv", index = False)
AG_news_df_sampled_augmented = pd.read_csv("AG_News_Dataset/AG_news_df_preprocessed.csv",)



############## TREC DATASET ##############

trec_df_tagged = augment_data(trec_df, remove_dates_ = False)
trec_df_tagged.to_csv("TREC_Dataset/trec_df_tagged.csv", index = False)

