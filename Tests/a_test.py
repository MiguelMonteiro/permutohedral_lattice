import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import lattice_filter_op_loader

module = lattice_filter_op_loader.module


def gaussian_blur(image, theta, normalize=True):
    x, y, c = image.shape
    I = np.zeros(shape=(x, y, c))
    for i in range(x):
        print(i)
        for j in range(y):
            norm_const=0
            for i_ in range(x):
                for j_ in range(y):
                    if i == i_ and j == j_:
                        continue
                    else:
                        w = np.exp(-((i-i_)**2 + (j-j_)**2) / (2*theta**2))
                        I[i, j, :] += w * image[i_, j_, :]
                        norm_const += w
            if normalize:
                I[i, j, :] /= norm_const
    return I


def bilateral(image, theta_s, theta_b=None, normalize=True):
    shape = image.shape
    spatial_dims = shape[:-1]
    num_channels = shape[-1]
    num_super_pixels = int(np.prod(spatial_dims))
    image = image.flatten()
    I = np.zeros(shape=(np.prod(shape)))
    norm_const = np.zeros(num_super_pixels)
    for i in range(num_super_pixels):
        print(i)
        for j in range(i+1, num_super_pixels):
            divisor = 1
            ws = 0
            for sdim in spatial_dims:
                i_ = (i / divisor) % sdim
                j_ = (j / divisor) % sdim
                ws += (i_ - j_)**2
                divisor *= sdim
            ws = -ws / (2 * (theta_s ** 2))

            wb = 0
            if theta_b:
                for c in range(num_channels):
                    I_i = image[num_channels * i + c]
                    I_j = image[num_channels * j + c]
                    wb += (I_i - I_j)**2
                wb = -wb / (2 * (theta_b ** 2))

            w = np.exp(ws + wb)
            norm_const[i] += w
            norm_const[j] += w
            for c in range(num_channels):
                I[num_channels * i + c] += w * image[num_channels * j + c]
                I[num_channels * j + c] += w * image[num_channels * i + c]
        if normalize:
            for c in range(num_channels):
                I[num_channels * i + c] /= norm_const[i]
    I = np.reshape(I, newshape=shape)
    return I



shape = [1, 20, 20, 3]
np.random.seed(1)
image = np.array(np.uint8(255 * np.random.rand(30, 30, 3)))
#image = np.array(np.uint8(255 * np.ones(shape=(60, 60, 3))))

#diff = gaussian_blur(image, 10.0) - gaussian_blur_2(image, 10.0)
image = np.array(Image.open('Images/small_cat.jpeg'))

with tf.device('gpu:0'):
    tf_input_image = tf.expand_dims(tf.constant(image, dtype=tf.float32), axis=0)
    tf_reference_image = tf_input_image
    theta_alpha = tf.constant(8.0)
    theta_beta = tf.constant(100.0)
    theta_gamma = tf.constant(1.0)
    y = module.lattice_filter(tf_input_image,
                              tf_reference_image,
                              theta_alpha=theta_alpha,
                              theta_beta=theta_beta,
                              theta_gamma=theta_gamma,
                              bilateral=True)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:
        b = sess.run(tf.round(tf.squeeze(y)))

output_image = np.array(b, dtype=np.uint8)
im = Image.fromarray(output_image)
im.save('cat_lattice.bmp')

I = bilateral(image, 8.0, 100.0, True)
output_image = np.array(I, dtype=np.uint8)
im = Image.fromarray(output_image)
im.save('cat_exact.bmp')

print(np.max(b-I))
exit(1)

I = gaussian_blur(image, theta=10.0)
output_image = np.array(I, dtype=np.uint8)
im = Image.fromarray(output_image)
im.save('cat.bmp')





exit(1)

T = module.lattice_filter(tf.ones(shape=(1, 4, 4, 1)),
                          tf_reference_image,
                          theta_alpha=.1,
                          theta_beta=.1,
                          theta_gamma=1,
                          bilateral=False)

with tf.Session() as sess:
    b = sess.run(tf.round((tf.squeeze(T))))

print(b)
exit(1)


with tf.Session() as sess:
    output_image = sess.run(tf.round((tf.squeeze(y))))

output_image = np.array(output_image, dtype=np.uint8)
im = Image.fromarray(output_image)
im.save('cat.bmp')

with tf.Session() as sess:
    theory, numerical = tf.test.compute_gradient(
        x=theta_gamma,
        x_shape=(),
        y=y,
        y_shape=shape,
        delta=1e-6,
        x_init_value=np.array(1.0))

print theory
print numerical
print(theory.shape)
t = theory -numerical
print('Ehre')
print(np.max(t))
print(np.mean(t))


# values = np.logspace(-2, 0.5, base=10)
# m = []
# for theta_gamma in values:
#     y = module.bilateral(tf_input_image, tf_reference_image,
#                          theta_alpha=.5, theta_beta=.5, theta_gamma=theta_gamma, bilateral=False)
#     with tf.Session() as sess:
#         max_error = tf.test.compute_gradient_error(x=tf_input_image, x_shape=shape, y=y, y_shape=shape, delta=1e-3)
#     m.append(max_error)
#
# print(max_error)
#
# with tf.Session() as sess:
#     j = tf.test.compute_gradient(x=tf_input_image, x_shape=shape, y=y, y_shape=shape, delta=1e-3)
#
# diff = np.abs(j[0] - j[1])
# print(np.mean(diff[diff != 0]))
