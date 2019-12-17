import os
import nltk
import pandas as pd
import numpy as np

import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

"""
This file implement a pos_tag model
And this model will be used to classify the word in our dataset
"""

## VOCABULARY with 3194 tagged sentences from NLTK
TAGGED_VOCABULARY = nltk.corpus.treebank.tagged_sents(tagset='universal')

def build_features(sent, index, label=True):
    '''
    This function creating features for each word in sentence
    It takes sentence and the word index as input
    Output features as a dict
    '''

    # As for the testing data we do not have labels (pos)
    if label:
        word = sent[index][0]
        label = sent[index][1]
        if index!=0:
            prev_word = sent[index-1][0]
        if index!=len(sent)-1:
            next_word = sent[index+1][0]
    else:
        word = sent[index]
        label = False
        if index!=0:
            prev_word = sent[index-1]
        if index!=len(sent)-1:
            next_word = sent[index+1]

    features = {
        'word':word,
        'is_first_word': int(index==0),
        'is_last_word':int(index==len(sent)-1),
        'prev_word':'' if index==0 else prev_word,
        'prev_word_last_1':prev_word[-1] if index!=0 else '',
        'prev_word_last_2':prev_word[-2:] if index!=0 else '',
        'next_word':'' if index==len(sent)-1 else next_word[0],
        'next_word_last_1':next_word[0][-1] if index!=len(sent)-1 else '',
        'next_word_last_2':next_word[0][-2:] if index!=len(sent)-1 else '',
        'is_numeric':int(word.isdigit()),
        'first_1':word[0],
        'first_2': word[:2],
        'first_3':word[:3],
        'first_4':word[:4],
        'last_1':word[-1],
        'last_2':word[-2:],
        'last_3':word[-3:],
        'last_4':word[-4:],
        'is_numeric': word.isdigit(),
        'word_has_hyphen': 1 if '-' in word else 0,
        'label': label if label else ''
         }
    
    return features

def get_data_label(sents, label=True):
    '''
    This function use build_features to get features 
    for each word in ench sents in a text
    '''
    
    features_list = list()
    for sent in sents:
        for index in range(len(sent)):
            features_list.append(build_features(sent, index, label))

    return features_list

def train_pos_tag():
    '''
    This function use TAGGED_VOCABULARY which is a labeled tag corpus in nltk
    to train a tag classification model
    It takes nothing as input
    Output trained model and a onehot encoder
    '''
    train_set, test_set = train_test_split(TAGGED_VOCABULARY,test_size=0.2,random_state=1234)
    train_features = get_data_label(train_set)
    test_features = get_data_label(test_set)

    train_df = pd.DataFrame(train_features)
    test_df = pd.DataFrame(test_features)

    features = [
        'word',
        'is_first_word',
        'is_last_word',
        'prev_word',
        'prev_word_last_1',
        'prev_word_last_2',
        'next_word',
    #     'next_word_last_1',
    #     'next_word_last_2',
        'is_numeric',
        'first_1',
        'first_2',
        'first_3',
        'first_4',
        'last_1',
        'last_2',
        'last_3',
        'last_4',
        'is_numeric',
        'word_has_hyphen'
    ]

    X_train = train_df[features].values
    Y_train = train_df.label
    # X_test = test_df[features].values
    # Y_test = test_df.label

    # encode word features by onehot encoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X_train)

    X_train = enc.transform(X_train)
    # X_test = enc.transform(X_test)

    # We use RandomForestClassifier as our classifier model
    # random forest is an improved model of Decision tree
    # It has better profermonce than Decision tree
    clf = RandomForestClassifier(
        n_estimators = 100,
        random_state=2019
    )

    clf.fit(X_train, Y_train)

    return clf, enc

if __name__ == "__main__":
    '''
    run the test while running this python file
    '''
    train_pos_tag()