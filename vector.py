import pandas as pd
import numpy as np

class Vector():
    '''
    This class create a vector for each text in our dataset
    It also provid a function for calculating tf
    '''
    def __init__(self, tokens, vocabulary):
        self.tokens = tokens
        self.vocabulary = vocabulary
        self.size = len(tokens)

    def __str__(self):
        return f"<Vector size={self.size}>"

    def __len__(self):
        '''
        return the size of vector: len(Vector)
        '''
        return self.size

    def tf(self):
        '''
        encode tokens by frequency
        return a frequency array
        '''
        encode_vec = np.zeros([len(self.vocabulary)], dtype='int16')
        join_set = set(self.tokens) & set(self.vocabulary.words)
        for token in self.tokens:
            if token in join_set:
                encode_vec[self.vocabulary.pos(token)] += 1

        return encode_vec/(len(join_set) + 1)

def test():
    '''
    testing the Vector class and its functions
    '''
    pass

if __name__ == "__main__":
    '''
    run the test while running this python file
    '''
    test()
        