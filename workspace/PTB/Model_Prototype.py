'''
Created on Jul 9, 2016

@author: mjchao
'''
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

class Learner(object):

    def __init__(self, word_to_id, lstm_size=200):
        self._word_to_id = word_to_id
        self._lstm_size = lstm_size
        self._session = tf.Session()

    def Train(self, data_iterator, batch_size=128, num_steps=128,
              max_grad_norm=5):
        input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", 
                                    [len(self._word_to_id), self._lstm_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)

        self._lstm = tf.nn.rnn_cell.BasicLSTMCell(self._lstm_size)
        softmax_W = tf.get_variable("softmax_W",
                                    [self._lstm_size, len(self._word_to_id)],
                                dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b",
                                [len(self._word_to_id)], dtype=tf.float32)

        initial_state = self._lstm.zero_state(batch_size, tf.float32)
        state = initial_state
        outputs = []
        for i in range(num_steps):
            if i > 0: tf.get_variable_scope().reuse_variables()
            cell_output, state = self._lstm(inputs[:, i, :], state)
            outputs.append(cell_output)
            
        output = tf.reshape(tf.concat(1, outputs), [-1, self._lstm_size])
        logits = tf.matmul(output, softmax_W) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
                                        [logits], 
                                        [tf.reshape(targets, [-1])], 
                                        [tf.ones(batch_size * num_steps)])
        cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        learning_rate = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(
                                        tf.gradients(cost, trainable_vars), 5)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(gradients, trainable_vars))


        sys.stdout.flush()
        with self._session.as_default():
            tf.initialize_all_variables().run()
            print "Evaluating initial state"
            sys.stdout.flush()
            numpy_state = initial_state.eval()
            total_cost = 0.0
            iters = 0
            for step, (x, y) in enumerate(data_iterator):
                print "Running iteration ", step
                sys.stdout.flush()
                numpy_state, batch_cost, _ = self._session.run(
                        [self._final_state, cost, train_op], 
                        feed_dict={initial_state: numpy_state,
                                   input_data: x, targets: y})
                total_cost += batch_cost
                iters += num_steps
                print "Perplexity: %.3f" %(np.exp(total_cost / iters))
                sys.stdout.flush()


def main():
    data_directory = "data"
    word_to_id = reader._build_vocab(os.path.join(data_directory, 
                                                  "ptb.train.txt"))
    train, cv, test, _ = reader.ptb_raw_data(data_directory)

    train_batch_size = 128
    train_num_steps = len(train) // train_batch_size - 1
    train_num_steps = 10
    ptb_iterator = reader.ptb_iterator(train, train_batch_size, train_num_steps)

    learner = Learner(word_to_id)
    learner.Train(ptb_iterator, train_batch_size, train_num_steps)

if __name__ == "__main__": main()