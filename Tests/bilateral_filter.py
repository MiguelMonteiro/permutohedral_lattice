# Test if bilateral filtering of  batch of images works (visual check required)
# GPU OP must be compiled with SPATIAL_DIMS=2 INPUT_CHANNELS=3 REFERENCE_CHANNELS=3
import tensorflow as tf
import numpy as np
from PIL import Image
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import lattice_filter_op_loader

module = lattice_filter_op_loader.module

theta_alpha = 8.0
theta_beta = 0.125

im = Image.open('Images/input.bmp')

tf_input_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)
tf_reference_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)

tf_input_batch = tf.stack([tf_input_image, tf_input_image])
tf_reference_batch = tf.stack([tf_reference_image, tf_reference_image])

output = module.lattice_filter(tf_input_batch, tf_reference_batch, theta_alpha=theta_alpha, theta_beta=theta_beta)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    o = np.round(sess.run(output) * 255).astype(np.uint8)

im = Image.fromarray(o[0])
im.save('Images/bilateral_out_1.bmp')
im = Image.fromarray(o[1])
im.save('Images/bilateral_out_2.bmp')
