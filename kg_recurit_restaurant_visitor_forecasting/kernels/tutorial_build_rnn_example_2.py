from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
# from tensorflow.nn import rnn, rnn_cell
import numpy as np


def simpleRNN(learning_rate=0.001,
              training_iters=100000,
              batch_size=256,
              display_step=100,
              n_input=28,
              n_steps=28,
              n_hidden=128,
              n_classes=10
              ):
    # 定义一些模型的参数
    '''
    To classify images using a reccurent neural network, we consider every image row as a sequence of pixels.
    Because MNIST image shape is 28*28px, we will then handle 28 sequences of 28 steps for every sample.
    '''
    # Parameters
    # learning_rate = 0.001
    # training_iters = 100000
    # batch_size = 256
    # display_step = 100

    # Network Parameters
    # n_input = 28 # MNIST data input (img shape: 28*28)
    # n_steps = 28 # timesteps
    # n_hidden = 128 # hidden layer num of features
    # n_classes = 10 # MNIST total classes (0-9 digits)
    # tf Graph input
    x = tf.placeholder("float32", [None, n_steps, n_input])
    # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
    y = tf.placeholder("float32", [None, n_classes])

    # Define weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # 首先创建一个CELL这里需要的一个参数是隐藏单元的个数n_hidden，
    # 在创建完成后对其进行初始化
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True)
    _state = lstm_cell.zero_state(batch_size, tf.float32)
    # 为了使得 原始数据的输入和模型匹配，我们对数据进行一系列变换，变换的结果如下
    a1 = tf.transpose(x, [1, 0, 2])
    a2 = tf.reshape(a1, [-1, n_input])
    a3 = tf.matmul(a2, weights['hidden']) + biases['hidden']
    a4 = tf.split(0, n_steps, a3)
    # 为了使得 原始数据的输入和模型匹配，我们对数据进行一系列变换，变换的结果如下

    # 或者前面解读RNN那篇
    outputs, states = tf.nn.rnn(lstm_cell, a4, initial_state=_state)
    a5 = tf.matmul(outputs[-1], weights['out']) + biases['out']
    # 定义cost，使用梯度下降求最优
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a5, y))
    # AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) # GradientDescent Optimizer
    correct_pred = tf.equal(tf.argmax(a5, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.initialize_all_variables()
    # 进行模型训练
    sess = tf.InteractiveSession()
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, })
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            print
            "Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print
    "Optimization Finished!"
    # Calculate accuracy for 256 mnist test images
    # 测试模型准确率
    test_len = batch_size
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(a5, 1), tf.argmax(y, 1))
    print
    "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label})
