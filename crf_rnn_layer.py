"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

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

import numpy as np
import tensorflow as tf
import lattice_filter_op_loader

module = lattice_filter_op_loader.module


def crf_rnn_layer(unaries, reference_image, num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations):
    spatial_ker_weights = tf.get_variable('spatial_ker_weights', shape=(num_classes, num_classes),
                                          initializer=tf.initializers.variance_scaling(distribution='uniform'))

    bilateral_ker_weights = tf.get_variable('bilateral_ker_weights', shape=(num_classes, num_classes),
                                            initializer=tf.initializers.variance_scaling(distribution='uniform'))

    compatibility_matrix = tf.get_variable('compatibility_matrix', shape=(num_classes, num_classes),
                                           initializer=tf.initializers.variance_scaling(distribution='uniform'))

    unaries_shape = unaries.get_shape()

    all_ones = np.ones(unaries_shape, dtype=np.float32)

    # Prepare filter normalization coefficients
    epsilon = 1e-8
    spatial_norm_vals = module.lattice_filter(all_ones, reference_image, bilateral=False, theta_gamma=theta_gamma) + epsilon
    bilateral_norm_vals = module.lattice_filter(all_ones, reference_image, bilateral=True, theta_alpha=theta_alpha,
                                                theta_beta=theta_beta) + epsilon

    q_values = unaries
    for i in range(num_iterations):

        if num_classes == 1:
            softmax_out = tf.nn.sigmoid(q_values)
        else:
            softmax_out = tf.nn.softmax(q_values, dim=-1)

        # Spatial filtering
        spatial_out = module.lattice_filter(softmax_out, reference_image, bilateral=False, theta_gamma=theta_gamma)

        spatial_out = spatial_out / spatial_norm_vals

        # Bilateral filtering
        bilateral_out = module.lattice_filter(softmax_out, reference_image, bilateral=True, theta_alpha=theta_alpha,
                                              theta_beta=theta_beta)

        bilateral_out = bilateral_out / bilateral_norm_vals

        # Weighting filter outputs
        a = tf.matmul(spatial_ker_weights, tf.reshape(tf.transpose(spatial_out), (num_classes, -1)))
        b = tf.matmul(bilateral_ker_weights, tf.reshape(tf.transpose(bilateral_out), (num_classes, -1)))
        message_passing = a + b

        #message_passing = tf.matmul(spatial_ker_weights, tf.reshape(spatial_out, (num_classes, -1))) + \
        #                  tf.matmul(bilateral_ker_weights, tf.reshape(bilateral_out, (num_classes, -1)))

        # Compatibility transform
        pairwise = tf.matmul(compatibility_matrix, message_passing)

        # Adding unary potentials
        pairwise = tf.reshape(tf.transpose(pairwise), unaries_shape)
        #pairwise = tf.reshape(pairwise, unaries_shape)
        q_values = unaries - pairwise

    return q_values, bilateral_out, spatial_out, pairwise
