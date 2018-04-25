import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from src import util


def bn_layer(input_d, output_d, input=None):
    mean, var = tf.nn.moments(input, [0])
    scale = tf.Variable(tf.ones([input_d]))
    beta = tf.Variable(tf.zeros([output_d]))
    bn = tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3)
    return tf.identity(bn, name="bn")


def base_layer(input_d, output_d, suffix, input=None):
    w = tf.Variable(np.random.normal(size=(input_d, output_d)).astype(np.float32))
    b = tf.Variable(tf.zeros([output_d]))
    z = tf.matmul(input, w) + b
    tf.summary.histogram('mlp_w_' + suffix, w)
    tf.summary.histogram('mlp_b_' + suffix, b)
    return tf.identity(z, name="layer")


def mlp(x, input_d, output_d):
    with tf.name_scope("mlp"):
        hidden_d = int(input_d / 2)
        # layer = tf.nn.relu(base_layer(input_d, hidden_d, 'hid', input=x))
        layer = base_layer(input_d, hidden_d, 'hid', input=x)
        layer = bn_layer(hidden_d, hidden_d, input=layer)
        layer = base_layer(hidden_d, output_d, 'out', input=layer)
    return layer


def lstm(ts_x, time_steps, num_units):
    with tf.name_scope("lstm"):
        ts_x_reshaped = tf.stack(tf.unstack(value=ts_x, num=time_steps, axis=1, name='input_steps'), axis=0)
        rnn_cell = rnn.BasicLSTMCell(num_units)
        outputs, last_states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=ts_x_reshaped,
                                       time_major=True, parallel_iterations=4, dtype="float32")
        # outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=last_states, initial_state=last_states,
        #                   time_major=True, parallel_iterations=4, dtype="float32")
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=ts_x_reshaped,
                                                   time_major=False, sequence_length=[time_steps])
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=rnn_cell, helper=helper, initial_state=last_states)
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=True)

    lstm_kernel, lstm_bias = rnn_cell.variables
    tf.summary.histogram('lstm_kernel', lstm_kernel)
    tf.summary.histogram('lstm_bias', lstm_bias)
    return outputs[-1]


def conv2d(x, W, b, suffix, strides=1):
    # Conv2D wrapper, with bias and relu activation
    # Given an input tensor of shape [batch, in_height, in_width, in_channels]
    # and a filter / kernel tensor
    # of shape [filter_height, filter_width, in_channels, out_channels]
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    tf.summary.histogram('conv_w_' + suffix, W)
    tf.summary.histogram('conv_b_' + suffix, b)
    return tf.nn.relu(x, name='conv2d')


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv(im_x, image_d, image_ch, out_d, dropout):
    weights = {
        # 2x2 conv, ch input, 4 outputs
        'conv_1': tf.Variable(tf.random_normal([2, 2, image_ch, 4]), name='conv_w_1'),
        # fully connected, 2*2*4 inputs, 32 outputs
        'dense_1': tf.Variable(tf.random_normal([2*2*4, 32]), name='conv_w_dense_1'),
        # 32 inputs, out_d outputs (class prediction)
        'out': tf.Variable(tf.random_normal([32, out_d]), name='conv_w_out')
    }

    biases = {
        'conv_1': tf.Variable(tf.random_normal([4]), name='conv_b_1'),
        'dense_1': tf.Variable(tf.random_normal([32]), name='conv_b_dense_1'),
        'out': tf.Variable(tf.random_normal([out_d]), name='conv_b_out')
    }
    with tf.name_scope("conv"):
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(im_x, shape=[-1, image_d, image_d, image_ch])

        # Convolution Layer
        conv1 = conv2d(x, weights['conv_1'], biases['conv_1'], '1')
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.contrib.layers.flatten(conv1)
        fc1 = tf.add(tf.matmul(fc1, weights['dense_1']), biases['dense_1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Data parameters
data_size = 10000
train_size = int(data_size * 0.8)
batch_size = 100
image_d = 4  # image is 4 x 4
image_ch = 3  # 3 image channels (e.g. R G B)
ts_input_d = 4  # input time series (4 time steps)
output_d = 3
cnx_d = 3  # context non recurrent features

# Model parameters
lstm_out_d = 3
conv_out_d = 8

context = np.random.rand(data_size, cnx_d)
ts_input = np.random.rand(data_size, ts_input_d)
image = np.random.rand(data_size, image_ch * image_d * image_d)
# set average of context, time series, and image as 3-d output
label = np.concatenate((image.mean(axis=1, keepdims=True),
                        context.mean(axis=1,  keepdims=True),
                        ts_input.mean(axis=1,  keepdims=True)), axis=1)

cnx_test = context[train_size:data_size, :]
ts_input_test = util.row_to_matrix(ts_input[train_size:data_size, :], split_count=ts_input_d)
image_test = image[train_size:data_size, :]
label_test = label[train_size:data_size, :]

cnx_x = tf.placeholder(tf.float32, (None, cnx_d), name='cnx_x')
ts_x = tf.placeholder(tf.float32, (None, ts_input_d, 1), name='ts_x')
im_x = tf.placeholder(tf.float32, (None, image_d * image_d * image_ch), name='image_x')
y = tf.placeholder(tf.float32, (None, output_d), name='ts_y')

lstm_out = lstm(ts_x, ts_input_d, lstm_out_d)
conv_out = conv(im_x, image_d, image_ch, conv_out_d, dropout=0.9)
mlp_x = tf.concat([conv_out, lstm_out, cnx_x], axis=1, name='mlp_x')
mlp_out = mlp(mlp_x, lstm_out_d + conv_out_d + cnx_d, output_d)

loss_function = tf.losses.mean_squared_error(labels=y, predictions=mlp_out)
train_step = tf.train.AdamOptimizer(learning_rate=0.02).minimize(loss_function)

# get tensor-flow session
session = tf.Session()

# summary writer
tf.summary.scalar('MSE error', loss_function)
summary_all = tf.summary.merge_all() # merge all summaries

summary_writer = tf.summary.FileWriter('logs/hybrid/run1')
summary_writer.add_graph(session.graph)

# initialize session variables
session.run(tf.global_variables_initializer())

for i in range(0, 1000):
    sample_idx = np.random.randint(train_size, size=batch_size)
    cnx_sample = context[sample_idx, :]
    ts_input_sample = util.row_to_matrix(ts_input[sample_idx, :], split_count=ts_input_d)
    image_sample = image[sample_idx, :]
    label_sample = label[sample_idx, :]

    # Run train step and the summary
    summary, _ = session.run([summary_all, train_step], feed_dict={
        cnx_x: cnx_sample, ts_x: ts_input_sample, im_x: image_sample, y: label_sample})

    summary_writer.add_summary(summary, i)

    if i % 50 == 0:
        loss = session.run(loss_function, feed_dict={
            cnx_x: cnx_test, ts_x: ts_input_test, im_x: image_test, y: label_test})
        print(i, " Loss ", loss)




