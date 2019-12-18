import nltk

# NLTK stoplist with 3136 words
STOPLIST = set(nltk.corpus.stopwords.words())

class Tokenizer():
    '''
    Used to split text
    Including two main function tokenize and sent_tokenize
    tokenize split text into words list
    sent_tokenize split text into sents list
    '''

    def __init__(self, stop_words=False, lower=False, eng=False):
        
        self.stop_words = stop_words
        self.lower = lower

    def tokenize(self, text):
        '''
        tokenize the text into tokens list
        take string as input
        output words' list
        '''

        if self.lower:
            text = text.lower()

        token_list = text.split()

        if self.stop_words:
            token_list_without_stop = list()
            for token in token_list:
                if token not in STOPLIST:
                    token_list_without_stop.append(token)

            return token_list_without_stop

        return token_list

    def sent_tokenize(self, text):
        '''
        A sample implementation of sentence tokenize
        Using . ? and ! to samply seperate text data
        take text as input, output a sentence list
        '''

        list_sent = list()
        for sent in text.split('.'):
            if '?' in sent:
                for sub_sent in sent.split('?'):
                    if '!' in sub_sent:
                        for sub_sub_sent in sent.split('!'):
                            list_sent.append(sub_sub_sent.split())
                    else:
                        list_sent.append(sub_sent.split())
            elif '!' in sent:
                for sub2_sent in sent.split('!'):
                    list_sent.append(sub2_sent.split())
            elif sent:
                list_sent.append(sent.split())
            else:
                pass

        return list_sent

def test():
    '''
    testing function
    '''
    tokenizer = Tokenizer(lower=True)
    test_sent = "God is Great! I won a lottery."
    sent_list = tokenizer.sent_tokenize(test_sent)
    print(sent_list)

    for sent in sent_list:
        print(tokenizer.tokenize(sent))

if __name__ == "__main__":
    test()