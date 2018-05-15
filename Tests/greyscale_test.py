import tensorflow as tf
import numpy as np
from PIL import Image
import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lattice_filter_op_loader

module = lattice_filter_op_loader.module

theta_alpha = 8.0
theta_beta = 0.125

im = Image.open('Images/input.bmp')

rgb = np.array(im)
grey = np.array(0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]).astype(np.uint8)

grey = np.expand_dims(np.expand_dims(grey, axis=0), axis=-1)

tf_input_image = tf.constant(grey/255.0, dtype=tf.float32)
tf_reference_image = tf.constant(np.expand_dims(rgb/255.0, axis=0), dtype=tf.float32)

output = module.lattice_filter(tf_input_image, tf_reference_image, theta_alpha=theta_alpha, theta_beta=theta_beta)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    o = np.round(sess.run(output) * 255).astype(np.uint8)


im = Image.fromarray(np.squeeze(grey))
im.save('Images/gray_original.bmp')

im = Image.fromarray(np.squeeze(o))
im.save('Images/filtered_grey.bmp')
