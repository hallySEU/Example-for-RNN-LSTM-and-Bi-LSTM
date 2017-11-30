#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ rnn-lstm-bi_lstm.py
 Author @ huangjunheng
 Create date @ 2017-11-13 16:02:27
 Description @ rnn, lstm and bi-lstm model based tensorflow
"""

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

# 定义训练参数
learning_rate = 0.001
training_steps = 10000
display_steps = 200
batch_size = 128

# 定义模型参数
input_size = 28
time_steps = 28
num_hidden = 128
num_class = 10

X = tf.placeholder(tf.float32, [None, time_steps, input_size])
Y = tf.placeholder(tf.float32, [None, num_class])

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_class]))
}

# 针对双向LSTM
bi_weights = {
    'out': tf.Variable(tf.random_normal([num_hidden * 2, num_class]))
}

biases = {
    'out': tf.Variable(tf.random_normal([num_class]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, time_steps, axis=1)

    # Define a rnn cell with tensorflow
    rnn_cell = rnn.BasicRNNCell(num_hidden)

    # Get rnn cell output
    # outputs is a list, len(outputs) is time_steps.
    # outputs[-1].shape = (batch_size, num_hidden)
    outputs, state = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def LSTM(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, time_steps, axis=1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden)

    # Get lstm cell output
    outputs, state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def BiLSTM(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, time_steps, axis=1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden)
    # Backward direction cell: 后向（反向）传递，就是将输入序列倒置即可
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    # output.shape = (time_step, batch_size, num_hidden * 2)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# 选择使用不同的cell单元
# logits = RNN(X, weights, biases)
logits = LSTM(X, weights, biases)
# logits = BiLSTM(X, bi_weights, biases)


y_ = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)
init = tf.global_variables_initializer()

# 评估
correct_pred = tf.equal(tf.argmax(y_, axis=1), tf.argmax(Y, axis=1)) # True, False 矩阵
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # True, False 矩阵 => 1, 0 => 求均值

with tf.Session() as session:
    session.run(init)

    for step in range(1, training_steps + 1):
        # batch_x.shape = [128, 28*28]
        # batch_y.shape = [128, 10]
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape(batch_size, time_steps, input_size)
        batch_y = batch_y.reshape(batch_size, num_class)
        session.run(train, feed_dict={X: batch_x, Y: batch_y})

        # run 评估
        if step % display_steps == 0 or step == 1:
            loss, acc = session.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print 'Step %d: loss=%f, acc=%f' % (step, loss, acc)

    print 'Training Over.'

    # 测试
    test_len = 200
    test_X = mnist.test.images[:test_len].reshape((test_len, time_steps, input_size))
    test_Y = mnist.test.labels[:test_len] # test_y.shape = [128, 10]

    test_acc = session.run(accuracy, feed_dict={X: test_X, Y: test_Y})
    print 'Testing Accuray:', test_acc


