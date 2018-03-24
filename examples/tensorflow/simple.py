import tensorflow as tf

sess = tf.Session()
# create a computation graph for a * b + 2
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
k = tf.placeholder(2)
multiply = tf.multiply(a, b)
add = tf.add(multiply, k)

print(sess.run(add, feed_dict={a: 2.5, b: 2.0}))
print(sess.run([multiply, add], feed_dict={a: 2.5, b: 2.0}))

