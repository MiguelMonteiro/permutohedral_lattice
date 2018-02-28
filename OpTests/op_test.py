import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
import lattice_filter_op_loader

module = lattice_filter_op_loader.module

theta_alpha = 20.0
theta_beta = 0.125

shape=[1200, 800, 3]
im = Image.open("../Images/input.bmp")

tf_input_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)
tf_reference_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)

tf_input_batch = tf.stack([tf_input_image, tf_input_image])
tf_reference_batch = tf.stack([tf_reference_image, tf_reference_image])

output = module.lattice_filter(tf_input_batch, tf_reference_batch, theta_alpha=theta_alpha, theta_beta=theta_beta)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    o = np.round(sess.run(output) * 255).astype(np.uint8)

im = Image.fromarray(o[0])
im.save("z_first_one.bmp")
im = Image.fromarray(o[1])
im.save("z_second_one.bmp")

#

shape = [20, 20, 3]
np.random.seed(1)

image = np.uint8(255 * np.random.rand(20, 20, 3)) / 255.0

tf_input_image = tf.constant(image, dtype=tf.float64)
tf_reference_image = tf.constant(image, dtype=tf.float64)


y = module.bilateral(tf_input_image, tf_reference_image,
                     theta_alpha=.5, theta_beta=.5, theta_gamma=0.5, bilateral=False)
with tf.Session() as sess:
    max_error = tf.test.compute_gradient_error(x=tf_input_image, x_shape=shape, y=y, y_shape=shape, delta=1e-3)



#values = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 2.0, 5.0, 10.0, 100.0]
values = np.logspace(-2, 0.5, base=10)

m = []
for theta_gamma in values:
    y = module.bilateral(tf_input_image, tf_reference_image,
                     theta_alpha=.5, theta_beta=.5, theta_gamma=theta_gamma, bilateral=False)
    with tf.Session() as sess:
        max_error = tf.test.compute_gradient_error(x=tf_input_image, x_shape=shape, y=y, y_shape=shape, delta=1e-3)
    m.append(max_error)


print(max_error)

with tf.Session() as sess:
    j = tf.test.compute_gradient(x=tf_input_image, x_shape=shape, y=y, y_shape=shape, delta=1e-3)

diff = np.abs(j[0]-j[1])
print(np.mean(diff[diff!=0]))



#im = Image.fromarray(o)
#im.save("this_is_it.bmp")



