import tensorflow as tf
import numpy as np
from PIL import Image

im = Image.open("../input.bmp")

module = tf.load_op_library('./bilateral.so')

tf_input_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)
tf_reference_image = tf.constant(np.array(im)/255.0, dtype=tf.float32)

output = module.bilateral(tf_input_image, tf_reference_image)
with tf.Session() as sess:
    o = sess.run(output)

print(o)