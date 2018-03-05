import tensorflow as tf
import numpy as np
from PIL import Image
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import lattice_filter_op_loader
from crf_rnn_layer import crf_rnn_layer

module = lattice_filter_op_loader.module

rgb = np.array(Image.open('Images/input.bmp'))
grey = np.array(0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]).astype(np.uint8)
grey = np.expand_dims(np.stack([grey, 255-grey], axis=-1), axis=0)

unaries = tf.Variable(grey / 255.0, dtype=tf.float32)
reference_image = tf.Variable(np.expand_dims(rgb, axis=0), dtype=tf.float32)

num_classes = 2
theta_alpha = 8.0
theta_beta = 8.0
theta_gamma = 1.0
num_iterations = 1
output = crf_rnn_layer(unaries, reference_image, num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations)
output = tf.nn.softmax(output, dim=-1)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    o = sess.run(output)

o = np.round(o * 255).astype(np.uint8)
o=np.squeeze(o)[:,:,1]
o[o < 128] = 0
o[o >= 128] = 255
im = Image.fromarray(np.squeeze(o))
im.save('z1_output.bmp')
