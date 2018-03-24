# create a network for z = x * w + b and y = sigmoid (z),
# trained using sigmoid cross entropy loss function (logit, label) = -mean(p(label)*log(logit))
import tensorflow as tf

# build the network
n_input_nodes = 2
n_output_nodes = 1
x = tf.placeholder(tf.float32, (None, n_input_nodes))
y = tf.placeholder(tf.float32, (None, n_output_nodes))
w = tf.Variable(tf.random_normal(n_input_nodes, n_output_nodes))
b = tf.Variable(tf.zeros(n_output_nodes))
z = x * w + b
output = tf.sigmoid(z)

# define the loss function to be minimized
loss_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y))

# define the optimization method
learning_rate = 0.02
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# data
x_train = [[0, 1], [0, 2], [1, 2], [2, 4], [1, 5], [2, 1], [2, 5]]
y_train = [1, 2, 3, 6, 6, 3, 7]  # y = x1 + x2
x_test = [[2, 2], [2, 3], [1, 3]]
y_test = [4, 5, 4]

# train the network
sess = tf.Session()
sess.run(optimizer, feed_dict={x: x_train, y: y_train})
