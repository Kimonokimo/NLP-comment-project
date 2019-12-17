import joblib

import numpy as np
import pandas as pd
from nltk.corpus import wordnet

from vocabulary import FileReader, Vocabulary
from tokenizer import Tokenizer
from vector import Vector
from dataclean import DataClean
from lemmatizer import Lemmatizer
from pos_tagger import build_features, get_data_label, train_pos_tag

from sklearn.linear_model import LogisticRegression

def train_test_split(dataset_array, labels, test_size=0.2):
    '''
    split np array dataset into train dataset and test dataset
    take dataset, labels, and test size(0-1) as input
    output train dataset and test dataset
    '''
    rand_list = list(range(dataset_array.shape[0]))
    # shuffle index list
    np.random.shuffle(rand_list)
    
    # get the number of test data
    split_number = int(test_size*dataset_array.shape[0])

    test_X = dataset_array[rand_list[:split_number]]
    test_Y = labels[rand_list[:split_number]]
    train_X = dataset_array[rand_list[split_number:]]
    train_Y = labels[rand_list[split_number:]]
    
    return train_X, train_Y, test_X, test_Y

def k_fold(train, k=5):
    '''
    split the np array dataset into k folds for cross validation
    take the train dataset as input
    output k folds with index
    '''
    fold_list = list()
    # get the length of dataset(number of text)
    length = train.shape[0]

    # make sure each fold has same size
    fold_len = length//k * k
    index_list = list(range(fold_len))
    for i in range(k):
        fold_list.append(index_list[i::5])

    return fold_list

def tag_map(treebank_tag):

    if treebank_tag[0] == 'J':
        return wordnet.ADJ
    elif treebank_tag[0] == 'V':
        return wordnet.VERB
    elif treebank_tag[0] == 'N':
        return wordnet.NOUN
    elif treebank_tag[0] == 'R':
        return wordnet.ADV
    else:
        return None

def pr(y_i, y, train_X):
    '''
    Calculating the average words encode value in each cats
    return the average result
    '''
    p = train_X[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


def get_mdl(y, train_X):
    '''
    Using LogisticRegression to pred result
    This model will return a Probability from 0 to 1
    return model and r
    '''
    r = np.log(pr(1,y,train_X) / pr(0,y,train_X))
    m = LogisticRegression(C=4, dual=False, solver='lbfgs')
    x_nb = train_X * r
    return m.fit(x_nb, y), r

def main():
    # read data from the raw data file
    file_reader = FileReader('train.csv')
    # get text from raw data
    train = file_reader.get_text()
    # get label and class from raw data
    labels, cla = file_reader.get_labels()

    # because all the basic function are implemented by ourself in this project
    # it will take a longer time to do the data preprocessing compare with nltk inbuild function
    # Therefore, we used only 10k data to test from alg here
    train_list = list(train)[:10000]
    # store data after cleaning
    print('Clean the data, remove special punctuations, numbers and abbreviations....')
    clean_list = list()
    cleaner = DataClean()
    for train_data in train_list:
        clean_list.append(cleaner.clean(train_data))
    print('Data clean done!')
    print('')
    tkn = Tokenizer()
    # train a random forest pos tagger classfication model
    print('Training a pos tagger classfication model....')
    pos_tagger, onehot_enc = train_pos_tag()
    print('Model training done!')
    print('')
    text_list = list()
    # split text into sents before pos_tag
    print('Start tokenizing and lemmatizing....')
    print('This step will take a few minutes')
    for clean_data in clean_list:
        sents = tkn.sent_tokenize(clean_data)
        text_list.append(sents)
    # features for pos_tag
    features = [
        'word','is_first_word','is_last_word','prev_word','prev_word_last_1',
        'prev_word_last_2','next_word','is_numeric','first_1','first_2','first_3',
        'first_4','last_1','last_2','last_3','last_4','is_numeric','word_has_hyphen'
    ]
    # init Lemmatizer
    lem = Lemmatizer()
    lem_texts = list()

    # tokenize, pos_tag and lammatize sentence by sentence
    for sents in text_list:
        word_features = pd.DataFrame(get_data_label(sents, label=False))
        # some data is empty
        if not word_features.empty:
            word_encode = word_features[features].values
            word_encode = onehot_enc.transform(word_encode)
            pred_pos = pos_tagger.predict(word_encode)

            lem_text = list()
            text = word_features.word
            for index in range(len(text)):
                lem_text.append(lem.lemmatize(text[index], tag_map(pred_pos[index])))

            lem_texts.append(lem_text)
        else:
            lem_texts.append([])
    print('Done!')
    print('')

    print('Start building the Vocabulary for our data....')
    voc = Vocabulary(lem_texts)
    voc.remove_stop_words()
    print('Done!')
    print('')

    print('Calculating idf....')
    print('It may take 3 minutes in this step')

    # get idf word dict from Vocabulary
    idf_reference = voc.idf()
    idf = np.zeros([len(voc)])

    for word in idf_reference:
        idf[voc.pos(word)] = idf_reference[word]
    print('idf done!')
    print('')

    # the tf-idf encode array
    data_array = np.zeros([len(lem_texts), len(voc)], dtype='int16')
    print('Calculating tf-idf....')
    for index, text in enumerate(lem_texts):

        vec = Vector(text, voc)
        data_array[index] = idf * vec.tf()
    print('Done!')
    print('')

    X, Y, test_X, test_Y = train_test_split(data_array, labels, test_size=0.5)
    
    # split the train set into 5 fold for Cross Validation
    # However Cross Validation is time consuming and not necessary in this project
    # We just use one val set to choose the best threshold
    k = 5
    fold_list = k_fold(X, k=k)
    one_size = len(fold_list[0])
    train_X = np.zeros([one_size * 4, test_X.shape[1]])
    train_Y = np.zeros([one_size * 4, 6], dtype='int64')

    # split train dataset and validation dataset
    for index, fold in enumerate(fold_list):
        if index != k - 1:
            train_X[index * one_size: index * one_size + one_size] = X[fold]
            train_Y[index * one_size: index * one_size + one_size] = Y[fold]
        else:
            val_X = X[fold]
            val_Y = Y[fold]

    preds = np.zeros((len(val_X), len(cla)))
    Pred_test = np.zeros((len(test_X), len(cla)))

    # We use LogisticRegression to train 6 models for each cat
    for index, cat in enumerate(cla):
        print('fit', cat)
        m,r = get_mdl(train_Y[:,index], train_X)
        preds[:,index] = m.predict_proba(val_X * r)[:,1]
        Pred_test[:,index] = m.predict_proba(test_X * r)[:,1]

    # searching for the best threshold
    threshold = [0.55, 0.6, 0.65, 0.7, 0.75]
    reslut_list = list()
    for t in threshold:
        sum_result = 0
        row, col = preds.shape
        pred_Y = np.zeros([row, col])
        for i in range(row):
            for j in range(col):
                if preds[i,j] >= t:
                    pred_Y[i,j] = 1
                else:
                    pred_Y[i,j] = 0

        # print out the pred result
        print(f'Validation set Accuracy (threshold={t}):')
        for index, cat in enumerate(cla):
            result = (pred_Y[:,index] == val_Y[:,index]).sum()/len(pred_Y)
            sum_result += result
            print(f'{cat} : {result}')
        print('')
        reslut_list.append(sum_result)
  
    # Using the best threshold pred test data set
    t = threshold[np.argmax(np.array(reslut_list))]
    print(f'The best threshold is {t}')
    row, col = Pred_test.shape
    pred_test_Y = np.zeros([row, col])
    for i in range(row):
        for j in range(col):
            if Pred_test[i,j] >= t:
                pred_test_Y[i,j] = 1
            else:
                pred_test_Y[i,j] = 0
    print('')
    print('#######################################')
    print('#######################################')
    print(f'Test set Accuracy (threshold={t}):')
    for index, cat in enumerate(cla):
        result = (pred_test_Y[:,index] == test_Y[:,index]).sum()/len(pred_test_Y)
        print(f'{cat} : {result}')

if __name__ == "__main__":
    '''
    run the test while running this python file
    '''
    main()