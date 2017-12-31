#!/usr/local/bin/python3
# Used https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py as reference
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import pandas as pd

import parseData

import pdb

tf.logging.set_verbosity(tf.logging.INFO)

# Custom LSTM Class
class LSTM():
    def __init__(self, input_tensor, hidden_size, num_classes):

        # Create LSTM cell
        # Static run of model
        self.cell = rnn.BasicLSTMCell(hidden_size)
        self.outputs, self.states = rnn.static_rnn(self.cell, input_tensor, dtype=tf.float32)

        # Get final LSTM state
        # self.final_state = tf.matmul(self.outputs[-1], w) + b

# Training Parameters
learning_rate = 0.001
training_steps = 30000
batch_size = 32
display_step = 200

# Network Parameters
num_hidden = 32 # hidden layer num of features

# Data
data = parseData.createDataframe()
dataDF, labelDF, nameEncoder, labelEncoder, labelBinarizer, maxNameLength = parseData.encodeNames(data)

data_train, data_test, label_train, label_test = parseData.create_train_test_set(dataDF, labelDF, 0.30)

input_tensor = tf.placeholder("float", [None, maxNameLength, 1])
unstack_input_tensor = tf.unstack(input_tensor, maxNameLength, 1)
labels = tf.placeholder("float", [None, len(labelEncoder.classes_)], name = "labels")

lstm = LSTM(unstack_input_tensor, num_hidden, len(labelEncoder.classes_))

# Fully connected layer
# Initialize weights and biases for cell
w = tf.Variable(tf.random_normal([num_hidden, len(labelEncoder.classes_)]))
b = tf.Variable(tf.random_normal([len(labelEncoder.classes_)]))
fc = tf.matmul(lstm.outputs[-1], w) + b

# Softmax
pred = tf.nn.softmax(fc)

# Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc, labels = labels))
train = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(loss)

# Evaluate model (with test logits, for dropout to be disabled)
test_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    # x = np.random.rand(8,1)
    # lstm_c, lstm_h, lstm_o = lstm.update(x.astype(np.float32))
    #
    # W = tf.get_variable("W", [12])

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train',sess.graph)
    test_writer = tf.summary.FileWriter('./test')

    sess.run(init)

    # Prep data for training
    train_indices = data_train.index.tolist()

    for step in range(1,training_steps+1):
        batch_indices = np.random.choice(train_indices, batch_size).tolist()
        batch_x = data_train.loc[batch_indices].as_matrix()
        batch_y = labelBinarizer.transform(label_train.loc[batch_indices].as_matrix())

        batch_x = batch_x.reshape((batch_size, maxNameLength, 1))

        sess.run(train, feed_dict={input_tensor: batch_x, labels: batch_y})

    # Test solution
    test_x = data_test.as_matrix()
    test_x = test_x.reshape((-1, maxNameLength, 1))
    test_y = labelBinarizer.transform(label_test.as_matrix())
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={input_tensor: test_x, labels: test_y}))

        # if step % display_step == 0:
        #     loss
    # loss = tf.nn.softmax(lstm.c_t)
