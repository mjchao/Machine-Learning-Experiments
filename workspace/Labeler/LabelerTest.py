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

    def testBuildDictionarySingleLabel(self):
        data = [(["one", "one", "one", "one", "one", "two"], [1]),
                (["one", "two"], [2]),
                (["three"], [3])]
        dictionary = Labeler.BuildWordIdDictionary(data)

        # From lowest to highest entropy, we have "three", "one", "two".
        # So the dictionary should have processed them in that order.
        self.assertEqual(dictionary.GetId("one"), 2)
        self.assertTrue(dictionary.GetId("two"), 3)
        self.assertTrue(dictionary.GetId("three"), 1)

    def testBuildDictionaryMultipleLabel(self):
        data = [(["one"], [1, 2, 3, 4, 5]),
                (["two"], [1, 2, 3]),
                (["three"], [4, 5])]
        dictionary = Labeler.BuildWordIdDictionary(data)
        self.assertEqual(dictionary.GetId("one"), 3)
        self.assertEqual(dictionary.GetId("two"), 2)
        self.assertEqual(dictionary.GetId("three"), 1)

    def testBuildFeatures(self):
        dictionary = Labeler.WordIdDictionary()
        dictionary.ProcessWord("one")
        dictionary.ProcessWord("two")
        dictionary.ProcessWord("three")
        data = [(["one", "one", "two", "two", "three"], [1, 2, 3, 4, 5]),
                (["one", "two", "three", "three", "three"], [1, 3, 5]),
                (["one", "two", "two", "two", "two"], [2, 4])]
        X, y, _ = Labeler.BuildFeaturesAndLabels(dictionary, data)
        print X
        np.testing.assert_array_equal(y, [[1, 1, 1, 1, 1],
                                          [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])

if __name__ == "__main__":
    unittest.main()