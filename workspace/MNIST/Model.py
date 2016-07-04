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

class SimpleLearner(object):
    """A simple digit recognizer that applies softmax regression.
    """
    def __init__(self):
        """Sets up the graph for softmax regression.
        """
        self.x_ = tf.placeholder(tf.float32, [None, WIDTH*HEIGHT])
        self.W_ = tf.Variable(tf.zeros([WIDTH*HEIGHT, 10]))
        self.b_ = tf.Variable(tf.zeros([10]))
        init = tf.initialize_all_variables()
        self.sess_ = tf.Session()
        self.sess_.run(init)

        # We want out prediction to be n rows of 10 probabilities (the
        # probability of being each digit). Therefore, we need to multiply
        # x (n x 784) by W (784 x 10) so that our output will have dimension
        # (n x 10).
        self.y_ = tf.nn.softmax(tf.matmul(self.x_, self.W_) + self.b_)

    def Train(self, train_data, batch_size=100, num_iters=1000):
        """Trains the model.
        
        Args:
            train_data: The training data. It must provide a function next_batch
                that allows us to get a training batch for batch SGD.
            batch_size: Size of the training batch.
            num_iters: Number of iterations for which to train.
        """
        # Use cross entropy loss to evaluate the performance of the softmax
        # regression.
        y_true = tf.placeholder(tf.float32, [None, 10])
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(
                        y_true*tf.log(self.y_), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
                                                    cross_entropy_loss)

        for _ in range(num_iters):
            batch_features, batch_labels = train_data.next_batch(batch_size)
            self.sess_.run([train_step], 
                     {self.x_: batch_features, y_true: batch_labels})

    def Predict(self, features):
        """Predicts what digit the given data represents.
        """
        prediction = tf.argmax(self.y_, 1)
        return self.sess_.run(prediction, {self.x_: features})

    def Test(self, test_data):
        """Tests the model on unseen test data.
        
        Args:
            test_data: The test data. It must provide field images and a field
                labels that give us the test features and labels.
        """
        y_true = tf.placeholder(tf.float32, [None, 10])
        correct_prediction = tf.equal(tf.argmax(self.y_, 1), 
                                      tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print "Test accuracy:", self.sess_.run(accuracy, 
                        {self.x_: test_data.images, y_true: test_data.labels})


def ThresholdPixels(pixels, threshold=0.0001):
    """Applies a binary threshold and converts all pixels to 0 or 1.
    
    Args:
        threshold: (float) Any pixels less than the threshold are set to 0
            and any pixels at least the threshold value are set to 1.
    """
    pixels[pixels < threshold] = 0
    pixels[pixels >= threshold] = 1 


def BuildSimpleLearner():
    """Trains a SimplerLearner model on the MNIST dataset
    
    Returns:
        learner: (SimpleLearner) The trained model.
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    learner = SimpleLearner()
    ThresholdPixels(mnist.train.images)
    learner.Train(mnist.train)
    ThresholdPixels(mnist.test.images)
    learner.Test(mnist.test)
    return learner


def DrawInputs():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    digit_pixels = mnist.train.images[3,:]
    from PIL import Image
    img = Image.new(mode="L", size=(28,28), color="black")
    for y in range(28):
        for x in range(28):
            if digit_pixels[y*28+x] > 0:
                img.putpixel((x, y), 255)
    img.save("digit.png", "png")

def main():
    BuildSimpleLearner()
    #DrawInputs()


if __name__ == "__main__": main()