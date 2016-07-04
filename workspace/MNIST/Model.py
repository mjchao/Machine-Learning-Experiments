'''
Created on Jul 3, 2016

@author: mjchao
'''
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

WIDTH = 28
HEIGHT = 28

class Learner(object):
    def __init__(self):
        self.x_ = tf.placeholder(tf.float32, [None, WIDTH*HEIGHT])

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == "__main__": main()