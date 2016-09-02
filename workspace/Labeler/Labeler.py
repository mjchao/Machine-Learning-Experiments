'''
Created on Sep 1, 2016

@author: mjchao
'''
import re
import sys
import nltk
import numpy as np

MAX_CATEGORIES = 250
RF_TREES = 10

class WordIdDictionary(object):
    """Maps words to integer IDs.
    """
    UNK = 0

    # A maximum size hyperparameter for the dictionary so that we don't
    # include too many useless words
    MAX_SIZE = 5000

    def __init__(self):
        self._word_to_id = {}
        self._id_to_word = {}
        self._next_id = 1

    def ProcessWord(self, word):
        """Adds the given word to the dictionary if it isn't already in it.

        Args:
            word: (string) A word to process
        """
        if word not in self._word_to_id:
            self._word_to_id[word] = self._next_id
            self._id_to_word[self._next_id] = word
            self._next_id += 1

    def GetId(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:
            return 0

    def GetWord(self, word_id):
        if word_id < self._next_id:
            return self._id_to_word[word_id]
        else:
            raise RuntimeError("Word ID out of range.")

    def Size(self):
        return self._next_id

def Tokenize(sentence):
    """Tokenizes a sentence into individual words. Punctuation is stripped.

    Args:
        sentence: (string) A sentence to tokenize

    Returns:
        tokens: (list of string) The individual tokens in the sentence.
    """
    tokens = []
    for token in nltk.word_tokenize(sentence):
        if bool(re.search('[A-za-z0-9]', token)):
            tokens.append(token.lower())

    return tokens

def ReadInput():
    """Reads the input and returns the data.

    Returns:
        train: (list of string questions and int list labels) The training data
        test: (list of string questions) The test data
    """
    first_line = sys.stdin.readline()
    T, E = [int(num) for num in first_line.split()]

    # Read training data
    train = []
    for _ in range(T):
        labels = [int(num) for num in sys.stdin.readline().split()]
        question = sys.stdin.readline()
        train.append((Tokenize(question), labels[1:]))

    test = []
    for _ in range(E):
        question = Tokenize(sys.stdin.readline())
        test.append(question)

    return train, test

def BuildWordIdDictionary(train_data):
    """Build a WordIdDictionary from words in training data with low entropy.

    Args:
        train_data: List of (tokens list, category list) tuples.

    Returns:
        dictionary: (WordIdDictionary) A dictionary where word IDs have been
            assigned in order of increasing entropy. (e.g. word 1 has lowest
            entropy, word 2 has second lowest, ...)
    """
    all_words = set()
    for datum in train_data:
        sentence = datum[0]
        for word in sentence:
            all_words.add(word)
    dictionary = WordIdDictionary()

    # Just add all words to dictionary. There's only about 16,000 of them
    # in the sample input.
    for word in all_words:
        dictionary.ProcessWord(word)
    return dictionary


class BayesianClassifier(object):
    """Applies a Bayesian model to the problem.
    """

    def __init__(self):
        pass

    def fit(self, dictionary, train_data):
        self._dictionary = dictionary
        self._weights = np.zeros((dictionary.Size(), MAX_CATEGORIES),
                                 dtype=np.float32)
        self._weights += 1 / float(dictionary.Size())
        self._word_counts = np.zeros(dictionary.Size(), dtype=np.float32)
        for datum in train_data:
            words = datum[0]
            labels = datum[1]
            for word in words:
                for label in labels:
                    self._weights[dictionary.GetId(word)][label] += 1
                    self._word_counts[dictionary.GetId(word)] += 1


    def GetWeights(self):
        return self._weights

    def predict(self, sentence):
        # compute P(c | w1, ..., wj)
        scores = np.zeros(MAX_CATEGORIES)
        for category in range(MAX_CATEGORIES):
            score = 1.0
            for word in sentence:
                word_id = self._dictionary.GetId(word)
                if word_id != 0:
                    score *= (self._weights[word_id][category] /
                              self._word_counts[word_id])
            scores[category] = score
        best_idxs = np.argsort(scores)[::-1]
        return best_idxs[:10]


def Train(dictionary, train_data):
    clf = BayesianClassifier()
    clf.fit(dictionary, train_data)
    return clf


def Test(dictionary, test_data, clf):
    for data in test_data:
        predictions = clf.predict(data)
        output_str = ""
        for pred in predictions:
            output_str += str(pred) + " "
        print output_str.strip()


def main():
    train, test = ReadInput()
    #print "Finished reading input"
    dictionary = BuildWordIdDictionary(train)
    #print "Finished building dictionary"
    clf = Train(dictionary, train)
    #print "Finished training"
    Test(dictionary, test, clf)

if __name__ == "__main__": main()