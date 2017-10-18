import tensorflow as tf
from tensorflow.contrib import rnn

#import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#network size
steps = 28
input_size = 28
output_size = 10
hidden_size = 128

#sample place holders
batch_x = tf.placeholder(tf.float32, [None, steps, input_size])
batch_y = tf.placeholder(tf.float32, [None, output_size])

#parameters weights
Why = tf.Variable(tf.random_normal([hidden_size, output_size], stddev = 0.1))
By = tf.Variable(tf.ones([1, output_size]))

#graph 
rnn_x = tf.unstack(batch_x, steps, axis = 1)
rnn_cell = rnn.BasicRNNCell(hidden_size)
outputs, state = rnn.static_rnn(rnn_cell, rnn_x, dtype = tf.float32)

#cost 
pred = tf.matmul(outputs[-1], Why) + By
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = batch_y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(batch_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

K = 100000

batch_size = 100
test_acc = 0 
for i in range(K):
    xx, yy = mnist.train.next_batch(batch_size)
    xx = xx.reshape(batch_size, steps, input_size)
    sess.run(train_step, feed_dict = {batch_x : xx, batch_y : yy})
    sample_cost = sess.run(cost, feed_dict = {batch_x : xx, batch_y : yy})
    sample_accuracy = sess.run(accuracy, feed_dict = {batch_x : xx, batch_y : yy})
    if (i % 100 == 0):
        print ("Iteration : ", i, " cost : ", sample_cost,  " accuracy : ", sample_accuracy, "test accu : ", test_acc)
    if (i % 600 == 0):
        test_acc = sess.run(accuracy, feed_dict = {batch_x : mnist.test.images.reshape([-1, steps, input_size]), batch_y : mnist.test.labels})



