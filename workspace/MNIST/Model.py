'''
Created on Jul 3, 2016

@author: mjchao
'''
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)