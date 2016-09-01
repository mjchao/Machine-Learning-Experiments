'''
Created on Sep 1, 2016

@author: mjchao
'''
import re
import sys
import nltk
import numpy as np
import sklearn.linear_model
import sklearn.multiclass
import sklearn.preprocessing

MAX_CATEGORIES = 250
RF_TREES = 20

class WordIdDictionary(object):
    """Maps words to integer IDs.
    """
    UNK = 0

    # A maximum size hyperparameter for the dictionary so that we don't
    # include too many useless words
    MAX_SIZE = 10000

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
    category_counts = [{} for _ in range(MAX_CATEGORIES)]
    all_words = set()
    for datum in train_data:
        sentence = datum[0]
        labels = datum[1]
        for word in sentence:
            all_words.add(word)
            for label in labels:
                if word in category_counts[label]:
                    category_counts[label][word] += 1
                else:
                    category_counts[label][word] = 1

    # Compute entropies
    all_words_list = list(all_words)
    entropies = np.zeros(len(all_words_list))
    for word_idx in range(len(all_words_list)):
        word = all_words_list[word_idx]
        occurrences = np.zeros(MAX_CATEGORIES)
        for category in range(MAX_CATEGORIES):
            if word in category_counts[category]:
                occurrences[category] += category_counts[category][word]

        probabilities = occurrences / np.float32(occurrences.sum())
        # Note: ignore numpy warning of divide by zero encountered in log2.
        # That's not possible because we apply np.where(probabilities > 0).
        # Also, the test cases pass
        entropy = np.sum(np.where(probabilities > 0,
                                  -probabilities * np.log2(probabilities), 0))
        entropies[word_idx] = entropy

    word_idxs_by_entropy = np.argsort(entropies)
    dictionary = WordIdDictionary()
    for i in range(min(WordIdDictionary.MAX_SIZE,
                       len(word_idxs_by_entropy))):
        dictionary.ProcessWord(all_words_list[word_idxs_by_entropy[i]])

    return dictionary


def BuildFeatures(dictionary, train_data):
    X_unprocessed = np.zeros((len(train_data), dictionary.Size()))
    for i in range(len(train_data)):
        sentence = train_data[i][0]
        for word in sentence:
            X_unprocessed[i][dictionary.GetId(word)] += 1

    X = X_unprocessed
    return X

def BuildLabels(train_data):
    mlb = sklearn.preprocessing.MultiLabelBinarizer()
    y = mlb.fit_transform([datum[1] for datum in train_data])
    return y, mlb

def BuildFeaturesAndLabels(dictionary, train_data):
    X = BuildFeatures(dictionary, train_data)
    y, mlb = BuildLabels(train_data)
    return X, y, mlb

def Train(dictionary, train_data):
    X, y, mlb = BuildFeaturesAndLabels(dictionary, train_data)
    clf = sklearn.multiclass.OneVsRestClassifier(sklearn.linear_model.SGDClassifier(loss="log"))
    clf.fit(X, y)
    return clf, mlb


def Test(dictionary, test_data, clf, mlb):
    X = BuildFeatures(dictionary, test_data)
    predictions = clf.predict_proba(X)
    for i in range(len(X)):
        probabilities = predictions[i]
        top_idxs = np.argsort(probabilities)
        output_str = ""
        for j in range(10):
            output_str += str(mlb.classes_[top_idxs[j]]) + " "
        print output_str.strip()


def main():
    print "Started"
    train, test = ReadInput()
    print "Finished reading input"
    dictionary = BuildWordIdDictionary(train)
    print "Finished building dictionary"
    clf, mlb = Train(dictionary, train)
    print "Finished testing"
    Test(dictionary, test, clf, mlb)

if __name__ == "__main__": main()