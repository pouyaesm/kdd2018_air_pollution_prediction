from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Nadam, Adam
import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np

class NN:

    @staticmethod
    def add_default_layers(model):
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        return model

    @staticmethod
    def keras_mlp(x_train, y_train, x_valid, y_valid, loss='mean_absolute_error'):
        model = Sequential()
        input_size = x_train.shape[1]
        middle_size = int(input_size / 2)
        bottle_neck_size = int(input_size / 3)
        output_size = y_train.shape[1]
        model.add(Dense(units=input_size, input_dim=input_size))
        NN.add_default_layers(model)
        # model.add(Dense(units=int(middle_size), kernel_initializer="uniform"))
        # NeuralNet.add_default_layers(model)
        # model.add(Dense(units=int(middle_size), kernel_initializer="uniform"))
        # NeuralNet.add_default_layers(model)
        model.add(Dense(units=int(middle_size)))
        NN.add_default_layers(model)
        # model.add(Dense(units=int(bottle_neck_size)))
        # NeuralNet.add_default_layers(model)
        # model.add(Dense(units=int(middle_size)))
        # NeuralNet.add_default_layers(model)
        model.add(Dense(units=output_size))

        model.compile(loss=loss, optimizer=Adam(lr=0.01))
        # model.fit(x_train, y_train, epochs=75, batch_size=5000,
        #           validation_data=(x_valid, y_valid), verbose=1)
        model.fit(x_train, y_train, epochs=100, batch_size=1000, verbose=1)
        return model

    # @staticmethod
    # def batch_norm(input_d, output_d, input=None):
    #     mean, var = tf.nn.moments(input, [0])
    #     scale = tf.Variable(tf.ones([input_d]))
    #     beta = tf.Variable(tf.zeros([output_d]))
    #     bn = tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3)
    #     return tf.identity(bn, name="bn")

    @staticmethod
    def batch_norm(input, scope, is_training: tf.placeholder, epsilon=0.001, decay=0.99, reuse=None):
        # Connect the boolean tensor-flow to the corresponding python boolean is_training
        return tf.cond(is_training,
                       true_fn=lambda: NN.batch_norm_layer(input, scope, True, epsilon, decay, reuse),
                       false_fn=lambda: NN.batch_norm_layer(input, scope, False, epsilon, decay, reuse=True))

    @staticmethod
    def batch_norm_layer(input, scope, is_training: bool, epsilon=0.001, decay=0.99, reuse=None):
        """
        Performs a batch normalization layer

        Args:
            input: input tensor
            scope: scope name
            is_training: python boolean value
            epsilon: the variance epsilon - a small float number to avoid dividing by 0
            decay: the moving average decay

        Returns:
            The ops of a batch normalization layer
        """
        with tf.variable_scope(scope, reuse=reuse):
            shape = input.get_shape().as_list()
            # gamma: a trainable scale factor
            gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
            # beta: a trainable shift value
            beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
            moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0),
                                         trainable=False)
            moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0),
                                         trainable=False)
            if is_training:
                # tf.nn.moments == Calculate the mean and the variance of the tensor x
                avg, var = tf.nn.moments(input, list(range(len(shape) - 1)))
                update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
                update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
                control_inputs = [update_moving_avg, update_moving_var]
            else:
                avg = moving_avg
                var = moving_var
                control_inputs = []
            with tf.control_dependencies(control_inputs):
                output = tf.nn.batch_normalization(input, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

        return output

    @staticmethod
    def linear(input_d, output_d, suffix, input=None):
        w = tf.Variable(np.random.normal(size=(input_d, output_d)).astype(np.float32))
        b = tf.Variable(tf.zeros([output_d]))
        z = tf.matmul(input, w) + b
        tf.summary.histogram('mlp_w_' + suffix, w)
        tf.summary.histogram('mlp_b_' + suffix, b)
        return tf.identity(z, name="layer")
