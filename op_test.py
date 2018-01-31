import tensorflow as tf
import numpy as np
from PIL import Image

im = Image.open("../input.bmp")

bilateral_module = tf.load_op_library('./bilateral.so')

input = tf.constant(np.array(im), dtype=tf.float32)
output = bilateral_module(input)
with tf.Session() as sess:
    o = sess.run(output)

print(o)