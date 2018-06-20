# Test if bilateral filtering of  batch of images works (visual check required)
# GPU OP must be compiled with SPATIAL_DIMS=2 INPUT_CHANNELS=3 REFERENCE_CHANNELS=3
import tensorflow as tf
import numpy as np
from PIL import Image
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import lattice_filter_op_loader

module = lattice_filter_op_loader.module

with tf.device('gpu:0'):
    theta_alpha = tf.Variable(8.0, dtype=tf.float32)
    theta_beta = tf.constant(0.125, dtype=tf.float32)
    theta_gamma = tf.constant(1.0, dtype=tf.float32)
    im = Image.open('Images/input.bmp')

    tf_input_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)
    tf_reference_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)

    tf_input_batch = tf.stack([tf_input_image, tf_input_image])
    tf_reference_batch = tf.stack([tf_reference_image, tf_reference_image])

    output = module.lattice_filter(tf_input_batch,
                                   tf_reference_batch,
                                   theta_alpha=theta_alpha,
                                   theta_beta=theta_beta,
                                   theta_gamma=theta_gamma,
                                   bilateral=True)
    init_op = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        _ = sess.run(init_op)
        o = np.round(sess.run(output) * 255).astype(np.uint8)

    im = Image.fromarray(o[0])
    im.save('t1.bmp')
    im = Image.fromarray(o[1])
    im.save('t2.bmp')