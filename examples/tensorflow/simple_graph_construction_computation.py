# create a computation graph for "a * b + 2", apply some inputs,
# and print the output of computational nodes
import tensorflow as tf

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
k = tf.constant(2, dtype=tf.float32)
multiply = tf.multiply(a, b)
add = tf.add(multiply, k)

# expected to be 2.5 * 2 + 2 = 7
print(sess.run(add, feed_dict={a: 2.5, b: 2.0}))

# expected to be [2.5 * 2, 2 * 2.5 + 2] = [5, 7]
print(sess.run([multiply, add], feed_dict={a: 2.5, b: 2.0}))

