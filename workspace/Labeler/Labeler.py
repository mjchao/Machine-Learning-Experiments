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

    # Limit on the number of bigrams to use (currently, this says to use all
    # of them).
    MAX_BIGRAMS = 20000

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
    bigrams = {}
    for datum in train_data:
        sentence = datum[0]
        for i in range(len(sentence)-1):
            all_words.add(sentence[i])
            if (sentence[i], sentence[i+1]) in bigrams:
                bigrams[(sentence[i], sentence[i+1])] += 1
            else:
                bigrams[(sentence[i], sentence[i+1])] = 1
        all_words.add(sentence[-1])
    dictionary = WordIdDictionary()

    # Just add all words to dictionary. There's only about 16,000 of them
    # in the sample input.
    for word in all_words:
        dictionary.ProcessWord(word)

    bigrams_list = list(bigrams.iteritems())
    bigrams_list.sort(key=lambda x: -x[1])
    for i in range(min(WordIdDictionary.MAX_BIGRAMS, len(bigrams_list))):
        dictionary.ProcessWord(bigrams_list[i][0][0] + " " + bigrams_list[i][0][1])

    return dictionary


class BagOfWordsClassifier(object):
    """Applies a Bag of Words model and optimizes for P(category | words).
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
            for i in range(len(words)):
                word = words[i]
                next_word = words[i+1] if i+1 < len(words) else None
                for label in labels:
                    self._weights[dictionary.GetId(word)][label] += 1
                    self._word_counts[dictionary.GetId(word)] += 1
                    if next_word is not None:
                        self._weights[dictionary.GetId(word + " " + next_word)][label] += 1
                        self._word_counts[dictionary.GetId(word + " " + next_word)] += 1


    def GetWeights(self):
        return self._weights

    def predict(self, sentence):
        # compute P(c | w1, ..., wj)
        scores = np.zeros(MAX_CATEGORIES)
        for category in range(MAX_CATEGORIES):
            score = 1.0
            for i in range(len(sentence)):
                word = sentence[i]
                word_id = self._dictionary.GetId(word)
                if word_id != 0:
                    score *= (self._weights[word_id][category] /
                              self._word_counts[word_id])

                next_word = sentence[i+1] if i+1 < len(sentence) else None
                if next_word is not None:
                    bigram = word + " " + next_word
                    bigram_id = self._dictionary.GetId(bigram)
                    if bigram_id != 0:
                        score *= (self._weights[bigram_id][category] /
                                  self._word_counts[bigram_id])
            scores[category] = score if score != 1.0 else 0.0
        #print np.sort(scores/scores.sum())[::-1][:10]
        best_idxs = np.argsort(scores)[::-1]
        return best_idxs[:10]

class HmmClassifier(object):
    """Applies an HMM model and optimizes for P(w2 | w1, c)*P(w3 | w2, c)...
    """
    def __init__(self):
        pass

def Train(dictionary, train_data):
    clf = BagOfWordsClassifier()
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