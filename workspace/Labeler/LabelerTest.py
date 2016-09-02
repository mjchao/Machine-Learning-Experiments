'''
Created on Sep 1, 2016

@author: mjchao
'''
import unittest
import numpy as np
import Labeler


class LabelerTest(unittest.TestCase):


    def testTokenize(self):
        sentence = "12 34 56"
        expected = ["12", "34", "56"]
        self.assertEqual(Labeler.Tokenize(sentence), expected)

        sentence = "12+ 56"
        expected = ["12+", "56"]
        self.assertEqual(Labeler.Tokenize(sentence), expected)

        sentence = "12!, 34?"
        expected = ["12", "34"]
        self.assertEqual(Labeler.Tokenize(sentence), expected)

        sentence = "The man's coat was red, and blue."
        expected = ["the", "man", "'s", "coat", "was", "red", "and", "blue"]
        self.assertEqual(Labeler.Tokenize(sentence), expected)

    def testBuildDictionary(self):
        data = [(["one", "one", "one", "one", "one", "two"], [1]),
                (["one", "two"], [2]),
                (["three"], [3])]
        dictionary = Labeler.BuildWordIdDictionary(data)

        self.assertTrue(dictionary.GetId("one") > 0)
        self.assertTrue(dictionary.GetId("two") > 0)
        self.assertTrue(dictionary.GetId("three") > 0)

    def testClassifier(self):
        clf = Labeler.BayesianClassifier()


if __name__ == "__main__":
    unittest.main()