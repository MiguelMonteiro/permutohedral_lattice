import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import lattice_filter_op_loader

module = lattice_filter_op_loader.module

shape = [1, 20, 20, 3]
np.random.seed(1)

image = np.uint8(255 * np.random.rand(1, 20, 20, 3)) / 255.0

tf_input_image = tf.constant(image, dtype=tf.float32)
tf_reference_image = tf_input_image


values = np.logspace(-2, 0.5, base=10)
m = []
for theta_gamma in values:
    y = module.lattice_filter(tf_input_image,
                              tf_reference_image,
                              theta_alpha=.5,
                              theta_beta=.5,
                              theta_gamma=theta_gamma,
                              bilateral=False)
    with tf.Session() as sess:
        max_error = tf.test.compute_gradient_error(x=tf_input_image, x_shape=shape, y=y, y_shape=shape, delta=1e-3)
    m.append(max_error)

print(m)
print(np.max(m))






