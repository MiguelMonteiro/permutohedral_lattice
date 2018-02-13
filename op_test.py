import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
import bilateral_grad

module = bilateral_grad.module

theta_alpha = 8.0
theta_beta = 0.125

shape=[1200,800,3]
im = Image.open("../input.bmp")

tf_input_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)
tf_reference_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)

output = module.bilateral(tf_input_image, tf_reference_image, theta_alpha=theta_alpha, theta_beta=theta_beta)
output_2 = module.bilateral(tf_input_image, tf_reference_image, theta_gamma=8.0, bilateral=False)
with tf.Session() as sess:
    o = sess.run(output) * 255
    o2 = sess.run(output_2) * 255
o = np.round(o).astype(np.uint8)
o2 = np.round(o2).astype(np.uint8)

im = Image.fromarray(o)
im.save("this_is_it.bmp")
im = Image.fromarray(o2)
im.save("this_is_it_2.bmp")

#

shape = [20, 20, 3]
np.random.seed(1)

image = np.random.rand(20,20,3) * 255

tf_input_image = tf.constant(image, dtype=tf.float32)
tf_reference_image = tf.constant(image, dtype=tf.float32)

y = module.bilateral(tf_input_image, tf_reference_image,
                     theta_alpha=8, theta_beta=.125, theta_gamma = 0.5, bilateral=True)


with tf.Session() as sess:
    max_error = tf.test.compute_gradient_error(x=tf_input_image, x_shape=shape, y=y, y_shape=shape, delta=1e-3)

print(max_error)

with tf.Session() as sess:
    j = tf.test.compute_gradient(x=tf_input_image, x_shape=shape, y=y, y_shape=shape, delta=1e-3)

diff = np.abs(j[0]-j[1])
print(np.mean(diff[diff!=0]))



#im = Image.fromarray(o)
#im.save("this_is_it.bmp")



