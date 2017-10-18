from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def forward(x):

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev = 0.1)) #a bunch of one dim filter
    b1 = tf.Variable(tf.zeros((1, 32)))# tensorflow. zeros nor.mal ((shape in []))

    z1 = tf.nn.conv2d(x_image, w1, strides = [1, 1, 1, 1], padding = 'SAME')
    h1 = tf.nn.relu(z1 + b1)
    h1 = tf.nn.max_pool(h1, ksize = [1,2,2,1], strides = [1, 1, 1, 1], padding = 'SAME')

#-1, 28 28, 32
    w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1))
    b2 = tf.Variable(tf.zeros((1, 64)))

    z2 = tf.nn.conv2d(h1, w2, strides = [1, 1, 1, 1], padding = 'SAME')
    h2 = tf.nn.relu(z2 + b2)
    h2 = tf.nn.max_pool(h2, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding = 'SAME')

#-1, 28, 28, 64
    w3 = tf.Variable(tf.truncated_normal((28 * 28 * 64, 1024), stddev = 0.1))
    b3 = tf.Variable(tf.ones((1, 1024)))

    h2_straight = tf.reshape(h2, [-1, 28 * 28 * 64])
    h3 = tf.nn.relu(tf.matmul(h2_straight, w3) + b3)

    w4 = tf.Variable(tf.truncated_normal((1024, 10), stddev = 0.1))
    b4 = tf.Variable(tf.ones((1,10)))

    y_hat = tf.matmul(h3, w4) + b4
    return y_hat

tf.logging.set_verbosity(tf.logging.INFO)
mnist = input_data.read_data_sets('/tmp/tmp/mnist', one_hot = True)
x = tf.placeholder(tf.float32, [None, 784]) #can we do it without dimension?
y = tf.placeholder(tf.float32, [None, 10])

y_hat = 0
cluster = tf.train.ClusterSpec({"worker": ["localhost:2222", "localhost:2223"], "ps": ["localhost:2221"]})
server = tf.train.Server(cluster, job_name="ps", task_index=0)
server.join()
"""
with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/ps:0", cluster=cluster)):
    y_hat = forward(x)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_hat))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#all the above is create graph..

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_hat,1)) , tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session(server.target)
sess.run(init)

saver = tf.train.Saver() #saver = tf.train.Saver()
#saver.restore(sess, './model/model20')
#rate = (sess.run(accuracy, feed_dict = {x: mnist.test.images, y:mnist.test.labels}))
rate = 0.
for i in range(50000):
    x_batch, y_batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict = {x : x_batch, y : y_batch})
    print ("iteration : ", i, "batch rate: ", sess.run(accuracy, feed_dict = {x : x_batch, y : y_batch}), "test rate : ", rate)
    if (i % 1000 == 999):
        saver.save(sess, 'model/model' + str(i))
        rate = (sess.run(accuracy, feed_dict = {x: mnist.test.images, y:mnist.test.labels}))
"""
