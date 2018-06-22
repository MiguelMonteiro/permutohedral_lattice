"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana , Miguel Monteiro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
import numpy as  np
from tensorflow.python.framework import ops
from os import path

module = tf.load_op_library(path.join(path.dirname(path.abspath(__file__)), 'lattice_filter.so'))

# assumes shape = (batch_size, sdim1, ..., sdim_m, num_channels)
#the output will have shape (batch_size, sdim1, ..., sdim_m, num_spatial_dims)
def get_spatial_position_vectors(image):
    shape = image.get_shape()[1:-1].as_list()
    num_spatial_dims = len(shape)
    # the shape of the image must be known, this is only done once when creating the graph so it's fine
    spatial_position_vectors = np.array(np.where(np.ones(shape=shape))).transpose().reshape(shape + [num_spatial_dims])
    return tf.constant(spatial_position_vectors, dtype=tf.float32)



def get_theta_grad(filter_input,
                   filter_output,
                   position_vectors,
                   filter_function,
                   theta):
    # 2 + num_input_channels * 2 linear time filters
    # (can be reduced to 2+2 if all input channels*position vectors are filtered together)

    position_vectors_squared = tf.pow(position_vectors, 2)

    #(1) These terms are input_channel independent.
    t1 = tf.reduce_sum(position_vectors_squared, axis=-1, keepdims=True)
    t2 = tf.reduce_sum(position_vectors * filter_function(position_vectors), axis=-1, keepdims=True)
    t3 = tf.reduce_sum(filter_function(position_vectors_squared), axis=-1, keepdims=True)

    # multiplying by output broadcasts to all the input channels (input_channels == output_channels ALWAYS)
    T1 = -filter_output * (t1 - 2 * t2 + t3)

    # (2) These terms are NOT input channel independent
    c1 = filter_output * t1
    input_channels = tf.split(filter_input, num_or_size_splits=filter_input.get_shape()[-1], axis=-1)
    c2 = tf.stack([tf.reduce_sum(position_vectors * filter_function(input_channel * position_vectors), axis=-1)
                   for input_channel in input_channels], axis=-1)
    c3 = tf.stack([tf.reduce_sum(filter_function(input_channel * position_vectors_squared), axis=-1)
                   for input_channel in input_channels], axis=-1)

    T2 = c1 - 2 * c2 + c3

    # (3)
    return tf.reduce_mean(T1 + T2) / tf.pow(theta, 3)



@ops.RegisterGradient("BilateralFilter")
def _lattice_filter_grad(op, grad):
    """ Gradients for the BilateralFilter op. We only need to calculate the gradients
    w.r.t. the first input (unaries) as we never need to backprop errors to the
    second input (reference image).

    Args:
    op: The `bilateral_filter` operation that we are differentiating.
    grad: Gradients with respect to the output of the `lattice_filter` op.

    Returns:
    Gradients with respect to the input of `bilateral_filter`.
    """
    input_image = op.inputs[0]
    reference_image = op.inputs[1]
    theta_spatial = op.inputs[2]
    theta_color = op.inputs[3]
    output = op.outputs[0]
    grad_vals = module.bilateral_fitler(grad, reference_image, theta_spatial, theta_color, reverse=True)

    def filter_function(filter_input):
        return module.bilateral_fitler(filter_input, reference_image, theta_spatial, theta_color)

    spatial_position_vectors = get_spatial_position_vectors(input_image)
    theta_spatial_grad = get_theta_grad(input_image, output, spatial_position_vectors, filter_function, theta_spatial)

    theta_color_grad = get_theta_grad(input_image, output, reference_image, filter_function, theta_color)

    return [grad_vals, tf.zeros_like(reference_image), theta_spatial_grad, theta_color_grad]

@ops.RegisterGradient("GaussianFilter")
def _lattice_filter_grad(op, grad):
    """ Gradients for the GaussianFilter op. We only need to calculate the gradients
    w.r.t. the first input (unaries) as we never need to backprop errors to the
    second input (reference image).

    Args:
    op: The `gaussian_filter` operation that we are differentiating.
    grad: Gradients with respect to the output of the `lattice_filter` op.

    Returns:
    Gradients with respect to the input of `gaussian_filter`.
    """
    input_image = op.inputs[0]
    theta_spatial = op.inputs[1]
    output = op.outputs[0]
    grad_vals = module.gaussian_filter(grad, theta_spatial=theta_spatial, reverse=True)

    return [grad_vals, 0]