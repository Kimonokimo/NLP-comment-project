import nltk

## install wordnet if not exist
try:
    from nltk.corpus import wordnet
except:
    nltk.download('wordnet')
    from nltk.corpus import wordnet

class Lemmatizer():
    '''
    The Lemmatizer class using wordnet dataset build a Lemmatizer
    '''

    def __init__(self):
        pass

    def lemmatize(self, word, pos=None):
        '''
        The lemmatize function take word and its pos as input
        output the result from wordnet.morphy(word, pos) if exist
        otherwise return the input word
        '''

        lemmas = wordnet.morphy(word, pos)
        if lemmas:
            return lemmas
        else:
            return word

def test():
    '''
    function for testing
    '''
    test_word_list = ['cats', 'running']
    lemmatizer = Lemmatizer()

    for word in test_word_list:
        print(f'{word} : {lemmatizer.lemmatize(word)}')


if __name__ == "__main__":
    test()