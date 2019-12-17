import nltk

import pandas as pd
import numpy as np

from dataclean import DataClean
from tokenizer import Tokenizer
from vector import Vector

# Vocabulary with 234,377 English words from NLTK
ENGLISH_VOCABULARY = set(w.lower() for w in nltk.corpus.words.words())

# NLTK stoplist with 3136 words
STOPLIST = set(nltk.corpus.stopwords.words())

class Vocabulary():
    
    """
    class to store the information of our dataset's vocabulary
    the vocabulary is build from our dataset without stopwords
    also used to findout the pos of word
    """
    def __init__(self, tokens_list):

        # building a reference dict for finding pos of word
        self.words_reference = {}
        # build a vocabulary set
        self.words = set()

        self.tokens_list = tokens_list
        
        for index, tokens in enumerate(self.tokens_list):
            for token in tokens:
                self.words.add(token)
        
        # change set to list, used to count the pos of words
        self.words = list(self.words)
        
        for index, word in enumerate(self.words):
            self.words_reference[word] = index

        self.size = len(self.words_reference.keys())
                        
    def __str__(self):
        return f'<Vocabulary size=this voc contain {self.size} words>'
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.words[index]
    
    def pos(self, word):
        return self.words_reference[word]

    def idf(self):
        '''
        Calculating idf for each word in our Vocabulary
        Worning, this function is little bit slow
        need improved, but still useable right now
        Maybe improve in the future work

        return a word reference list
        '''
        word_idf = dict()
        for word in self.words:
            word_idf[word] = 1

        for tokens in self.tokens_list:
            join_set = set(tokens) & set(self.words)
            for token in join_set:
                word_idf[token] += 1

        for word in word_idf:
            word_idf[word] = np.log(len(self.tokens_list)/word_idf[word])
        
        return word_idf

    def reset_vocabulary_by_english_vocabulary(self):
        
        new_words = set()
        self.words_reference = {}
        
        for word in self.words:
            if word in ENGLISH_VOCABULARY:
                new_words.add(word)
                
        new_words = list(new_words)
        
        for index, word in enumerate(new_words):
            self.words_reference[word] = index
            
        self.words = new_words
        self.size = len(self.words_reference.keys())

    def remove_stop_words(self):
        '''
        Remove all stop words in out vocabulary
        '''
        
        new_words = set()
        self.words_reference = {}
        
        for word in self.words:
            if word not in STOPLIST:
                new_words.add(word)
                
        new_words = list(new_words)
        
        for index, word in enumerate(new_words):
            self.words_reference[word] = index
            
        self.words = new_words
        self.size = len(self.words_reference.keys())

class FileReader():
    '''
    Create a file reader by pandas
    '''
    def __init__(self, path):
        self.train_csv = pd.read_csv(path)
        self.cla = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def get_text(self):
        '''
        get text from raw file
        '''
        return self.train_csv.comment_text.values

    def get_labels(self):
        '''
        get the training labels
        '''
        return self.train_csv[self.cla].values, self.cla

def test():
    '''
    testing the Vocabulary class and the FileReader class
    '''
    # check if FileReader works
    file_reader = FileReader('train.csv')
    train = file_reader.get_text()
    labels, cla = file_reader.get_labels()
    print(f'The first example of training data is : {train[0]}')
    print(f'The first example of label is : {labels[0]}')
    print(f'The class of our dataset includes : {cla}')

    # check DataClean class
    train_list = list(train)
    clean_list = list()
    cleaner = DataClean()
    for train_data in train_list:
        clean_list.append(cleaner.clean(train_data))

    # check Tokenizer class
    tkn = Tokenizer()
    tokens_list = list()

    for clean_data in clean_list:
        tokens_list.append(tkn.tokenize(clean_data))

    # check if Vocabulary is working
    vocabulary = Vocabulary(tokens_list)
    print(f'The length of vocabulary is : {len(vocabulary)}')

if __name__ == "__main__":
    '''
    run the test while running this python file
    '''
    test()
