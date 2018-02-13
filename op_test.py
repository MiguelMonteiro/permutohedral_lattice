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

image = np.random.rand(20,20,3)
tf_input_image = tf.constant(image, dtype=tf.float32)
tf_reference_image = tf.constant(image, dtype=tf.float32)


grad = module.bilateral(tf_input_image, tf_reference_image, reverse=True, theta_alpha=theta_alpha, theta_beta=theta_beta)

with tf.Session() as sess:
    out = gradient_checker.compute_gradient([tf_input_image, tf_reference_image], [shape, shape], grad, shape)

# We only need to compare gradients w.r.t. unaries
computed = out[0][0].flatten()
estimated = out[0][1].flatten()


mask = (computed != 0)
computed = computed[mask]
estimated = estimated[mask]
difference = computed - estimated

measure1 = np.mean(difference) / np.mean(computed)
measure2 = np.max(difference) / np.max(computed)

assert(measure1 <= 1e-3)
assert(measure2 <= 2e-2)



#im = Image.fromarray(o)
#im.save("this_is_it.bmp")



