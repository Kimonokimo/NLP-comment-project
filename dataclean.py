import re

class DataClean():
    '''
    Data Preprocessing
    As we can see the raw data contains a lot useless information
    includes: special punctuations, numbers and abbreviations
    The DataClean class uses re to clean punctuations and numbers,
    and replace abbreviations in english by their normal form
    '''
    def __init__(self):
        # findout all Punctuation exxcept ' . ? and !
        # we need . ? and ! to separate sentances
        self.pat_punctuation = re.compile(r'[^a-zA-Z \'.?!]+')
        # findout the words end with 's
        self.pat_is = re.compile("(|he|she|it|that|this|there|here)(\'s)", re.I)
        # findout the other wrds end with 's
        self.pat_s = re.compile("(?<=[a-zA-Z])\'s")
        # findout the words with 't
        self.pat_t = re.compile("(?<=[a-zA-Z])n\'t")
        # findout the word 
        self.pat_ll = re.compile("(?<=[a-zA-Z])\'ll")
        self.pat_ld = re.compile("(?<=[a-zA-Z])\'d")

        # findout I'm
        self.pat_m = re.compile("(?<=[i])\'m")
        self.pat_re = re.compile("(?<=[a-zA-Z])\'re")
        # findout word with 've
        self.pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    def clean(self, text):

        # replace 
        clean_text = self.pat_punctuation.sub(' ', text).strip().lower()
        # replace 's with is
        # he's -> he is
        clean_text = self.pat_is.sub(r"\1 is", clean_text)
        clean_text = self.pat_s.sub("", clean_text)
        # replace 't by not
        clean_text = self.pat_t.sub(" not", clean_text)
        # replace 'd by would
        clean_text = self.pat_ld.sub(" would", clean_text)
        # replace 'll by will
        clean_text = self.pat_ll.sub(" will", clean_text)
        # replace I'm by I am
        clean_text = self.pat_m.sub(" am", clean_text)
        clean_text = self.pat_re.sub(" are", clean_text)
        clean_text = self.pat_ve.sub(" have", clean_text)

        return clean_text

def test():
    # an example from out data file
    text = 'Hey man, I\'m really not trying to edit war. It\'s just that this guy is constantly \
    removing relevant information and talking to me through edits instead of my talk page. He \
    seems to care more about the formatting than the actual info.'

    dataclean = DataClean()
    print(dataclean.clean(text))

if __name__ == "__main__":
    '''
    run the test while running this python file
    '''
    test()