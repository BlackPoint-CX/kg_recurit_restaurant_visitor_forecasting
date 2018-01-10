# 知乎专栏: 如何用TensorFlow构建RNN？这里有一份极简的教程
# https://zhuanlan.zhihu.com/p/26646665

import numpy as np

# from __future__ import print_function, division

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000  # series总长度, 也就是有50000个二维值. 元素的总个数
truncated_backprop_length = 15  # 截断长度
state_size = 4  # 状态大小
num_classes = 2  # label num_classes个类
echo_step = 3  #
batch_size = 5  # 每次处理batch_size个样本
num_batches = total_series_length // batch_size // truncated_backprop_length


def generate_data():
    """
    生成随机的训练数据, 输入为一个随机的二元向量, 在echo_step个时间步之后, 可得到输入的'回声', 也就是输出
    :return:
    """
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)  # 将x的值向后推了echo_step步
    y[0:echo_step] = 0  # 把后推的前echo_step个值填0
    x = x.reshape((batch_size, -1))  # -1表示不知道具体值多少, 由Numpy自己推算进行填充
    y = y.reshape((batch_size, -1))
    return x, y


# [5,15] 二维数据, batch_size个样本, 每个样本长度为 truncated_backprop_length, 因为生成数据的时候特征长度为1, 所以截断长度为15
# 如果样本特征长度大于1, 例如[0.,1.,2.5,4] 是不是应该将二维数据输入转为三维? 例如[batch_size,echo_step,feature_size]
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])

batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])  # [5,15]

# state_size 代表隐藏状态的特征长度
init_state = tf.placeholder(tf.float32, [batch_size, state_size])  # [5,4]

# state_size+1是因为需要将state与输入特征相连,连接之后再与W乘.
W = tf.Variable(np.random.rand(state_size + 1, state_size), dtype=tf.float32)  # [5,4]

b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)  # [1,4]

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)  # [4,2]

b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)  # [1,2]

inputs_series = tf.unstack(batchX_placeholder, axis=1)

labels_series = tf.unstack(batchY_placeholder, axis=1)

current_state = init_state

state_series = []

for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])  # [5,1]
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # [5,5]
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # 激活函数

    state_series.append(next_state)  # 输出到state_series中
    current_state = next_state

logits_series = [tf.matmul(state, W2) + b2 for state in state_series]  # Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in
          zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x, y = generate_data()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss , train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
