import re

# regular transformation rules for nouns, verbs and adjectives
SUBSTITUTIONS = {
    "noun": [
        ("s", ""),
        ("ses", "s"),
        ("ves", "f"),
        ("xes", "x"),
        ("zes", "z"),
        ("ches", "ch"),
        ("shes", "sh"),
        ("men", "man"),
        ("ies", "y")
    ],
    "verb": [
        ("s", ""),
        ("ies", "y"),
        ("es", ""),
        ("es", "e"),
        ("ed", ""),
        ("ed", "e"),
        ("ing", ""),
        ("ing", "e")
    ],
    "adj": [
        ("er", ""),
        ("est", ""),
        ("er", "e"),
        ("est", "e"),
        ("ier", "y"),
        ("iest", "y")
    ]
}


def load_wordnet_words():
    # load all words in the WordNet Corpus as a lookup database
    words = []
    with open("wordnetWords.txt") as wn:
        for line in wn:
            words.extend(re.findall(r'(.+)\n', line))

    return set(words)


def load_irregular_forms(pos):
    # function that loads the irregular forms given the part of speech
    irregular_forms = {}
    if pos == "noun":
        file_path = "irregular_nouns.txt"
    elif pos == "verb":
        file_path = "irregular_verbs.txt"
    elif pos == "adj":
        file_path = "irregular_adjectives.txt"
    else:
        print("No irregular word database available.")

    with open(file_path, encoding='utf-16') as ir_n:
        for line in ir_n:
            single, plural = line.split()
            irregular_forms.update({tuple(line.split()[1:]): line.split()[0]})

    return irregular_forms


def apply_rules(token, pos):
    # apply transformation rules to tokens and return a list of possible forms
    form_list = []
    for old, new in SUBSTITUTIONS[pos]:
        if token.endswith(old):
            form_list.append(token[:-len(old)] + new)

    return form_list


def lemmatize(token, pos='noun'):
    # function that returns the lemmas of the input words

    # check if the input word is in wordnet database
    word_dic = load_wordnet_words()
    if token in word_dic:
        return token

    word_found = False

    # transform the input words and lookup the new words in the database
    possible_forms = apply_rules(token, pos)
    for form in possible_forms:
        if form in word_dic:
            return form

    # check if the words are irregular
    irregular_word_dic = load_irregular_forms(pos)
    for key in irregular_word_dic.keys():
        if token in key:
            return irregular_word_dic[token]

    # if the word and its transformation can not be found in the database
    # and the word can not be find in the irregular word list, return the
    # word itself
    if not word_found:
        return token


def test():
    # function for testing
    test_word_list = ['cats', 'running']

    for word in test_word_list:
        print(f'{word} : {lemmatize(word)}')


if __name__ == "__main__":
    test()
