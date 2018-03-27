# create a network for y = x * w + b,
# train the network to minimize mean squared error loss function via gradient descent
# evaluate a test set
# log the network and loss summary to be monitored in tensor-board

import tensorflow as tf

# parameters
n_input_nodes = 2  # input dimension
n_output_nodes = 1  # output dimension
epochs = 100  # number of weight updates

# build the network
with tf.name_scope('network'):
    x = tf.placeholder(tf.float32, (None, n_input_nodes), name='x')
    y = tf.placeholder(tf.float32, (None, n_output_nodes), name='y')
    w = tf.Variable(tf.random_normal([n_input_nodes, n_output_nodes]), name='weights')
    b = tf.Variable(tf.zeros([1, n_output_nodes]), name='biases')
    output = tf.matmul(x, w) + b

# define the loss function to be minimized (mean squared error)
with tf.name_scope('loss'):
    loss_function = tf.reduce_mean(tf.squared_difference(output, y))

# define the train step using gradient descent optimizer
learning_rate = 0.05
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# data
x_train = [[0, 1], [0, 2], [1, 2], [2, 4], [1, 5], [2, 1], [5, 2], [3, 1], [1, 0]]
y_train = [[0], [-1], [0], [-1], [-3], [2], [4], [3], [2]]  # y = x1 - x2 + 1
x_test = [[2, 2], [2, 3], [1, 3]]
y_test = [[1], [0], [-1]]

# create a session
sess = tf.Session()

# summaries of interest
tf.summary.scalar('error', loss_function)
tf.summary.histogram('weights', w)
tf.summary.histogram('biases', b)

# merge all summaries
summary_all = tf.summary.merge_all()
# summary writer
summary_writer = tf.summary.FileWriter('logs/simple_graph_monitoring/run3')
summary_writer.add_graph(sess.graph)

# initialize variables
sess.run(tf.global_variables_initializer())

# train the network
for e in range(epochs):
    summary, _ = sess.run([summary_all, train_step], feed_dict={x: x_train, y: y_train})
    summary_writer.add_summary(summary, e)

# output learned parameters (w, b)
print(w.name, w.eval(sess))
print(b.name, b.eval(sess))