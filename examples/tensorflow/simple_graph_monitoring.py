# create a network for y = x * w + b,
# train the network to minimize mean squared error loss function via gradient descent
# evaluate a test set

import tensorflow as tf

# parameters
n_input_nodes = 2  # input dimension
n_output_nodes = 1  # output dimension
epochs = 200  # number of weight updates

# build the network
x = tf.placeholder(tf.float32, (None, n_input_nodes), name='x')
y = tf.placeholder(tf.float32, (None, n_output_nodes), name='y')
w = tf.Variable(tf.random_normal([n_input_nodes, n_output_nodes]), name='weight')
b = tf.Variable(tf.zeros([1, n_output_nodes]), name='bias')
output = tf.matmul(x, w) + b

# define the loss function to be minimized (mean squared error)
loss_function = tf.reduce_mean(tf.squared_difference(output, y))
loss_summary = tf.summary.scalar('loss', loss_function)

# define the optimization method
learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# data
x_train = [[0, 1], [0, 2], [1, 2], [2, 4], [1, 5], [2, 1], [5, 2], [3, 1], [1, 0]]
y_train = [[-1], [-2], [-1], [-2], [-4], [1], [3], [2], [1]]  # y = x1 - x2
x_test = [[2, 2], [2, 3], [1, 3]]
y_test = [[0], [-1], [-2]]

# create a session
sess = tf.Session()

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# train the network
for e in range(epochs):
    sess.run([optimizer, loss_summary], feed_dict={x: x_train, y: y_train})

# summary writer
summary_writer = tf.summary.FileWriter('logs/', sess.graph)
tf.nn.max_pool()
# run summarizing operation

