import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lattice_filter_op_loader

module = lattice_filter_op_loader.module

if os.path.isfile('gray_original.bmp'):
    os.remove('gray_original.bmp')
if os.path.isfile('filtered_grey.bmp'):
    os.remove('filtered_grey.bmp')

theta_alpha = 8.0
theta_beta = 0.125

im = Image.open('Images/input.bmp')

rgb = np.array(im)
grey = np.array(0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]).astype(np.uint8)
im = Image.fromarray(grey)
im.save('gray_original.bmp')
grey = np.expand_dims(np.expand_dims(grey, axis=0), axis=-1)

tf_input_image = tf.constant(grey/255.0, dtype=tf.float32)
tf_reference_image = tf.constant(np.expand_dims(rgb/255.0, axis=0), dtype=tf.float32)

output = module.lattice_filter(tf_input_image, tf_reference_image, theta_alpha=theta_alpha, theta_beta=theta_beta)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    o = np.round(sess.run(output) * 255).astype(np.uint8)

#a = np.squeeze(o[0])
#rgb = np.stack((a,a,a), axis=-1)
im = Image.fromarray(np.squeeze(o[0]))
im.save('filtered_grey.bmp')
