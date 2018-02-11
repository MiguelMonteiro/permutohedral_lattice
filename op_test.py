import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
import bilateral_grad

module = bilateral_grad.module

theta_alpha = 1.0
theta_beta = 1.0
shape=[1200,800,3]
im = Image.open("../input.bmp")

tf_input_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)
tf_reference_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)

output = module.bilateral(tf_input_image, tf_reference_image, theta_alpha=theta_alpha, theta_beta=theta_beta)
with tf.Session() as sess:
    o = sess.run(output) * 255
o = np.round(o).astype(np.uint8)


#

shape = [20, 20, 3]

image = np.random.rand(20,20,3)
tf_input_image = tf.constant(image, dtype=tf.float32)
tf_reference_image = tf.constant(image, dtype=tf.float32)


grad = module.bilateral(tf_input_image, tf_reference_image, reverse=True, theta_alpha=theta_alpha, theta_beta=theta_beta)
point =  module.bilateral(tf_input_image, tf_reference_image, reverse=False, theta_alpha=theta_alpha, theta_beta=theta_beta)
epsilon= 1e-3
left = module.bilateral(tf_input_image - epsilon, tf_reference_image, reverse=False, theta_alpha=theta_alpha, theta_beta=theta_beta)
right = module.bilateral(tf_input_image + epsilon, tf_reference_image, reverse=False, theta_alpha=theta_alpha, theta_beta=theta_beta)


with tf.Session() as sess:
    g = sess.run(grad)
    p = sess.run(point)
    r = sess.run(right)
    l = sess.run(left)

(r - l) / (2 * epsilon)
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



