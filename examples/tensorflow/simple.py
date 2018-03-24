import tensorflow as tf

sess = tf.Session()
# create a computation graph for a * b + 2
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
k = tf.placeholder(2)
multiply = tf.multiply(a, b)
add = tf.add(multiply, k)



