import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
import nibabel as nib
import lattice_filter_op_loader

module = lattice_filter_op_loader.module


theta_alpha = 10.0
theta_beta = 0.5

im = nib.load("../Images/brain.nii")
image = np.expand_dims(im.get_data(), axis=-1).astype(np.float32)
image /= np.max(image)

tf_input_image = tf.constant(image, dtype=tf.float32)
tf_reference_image = tf.constant(image, dtype=tf.float32)

output = module.bilateral(tf_input_image, tf_reference_image, theta_alpha=theta_alpha, theta_beta=theta_beta, theta_gamma=1.0, bilateral=False)

with tf.Session() as sess:
    o = sess.run(output) * 3766

o = np.round(o).astype(np.int16)


i = im.get_data()
i[:] = np.squeeze(o)
nib.save(im, "filtered.nii")
