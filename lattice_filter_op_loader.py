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
from tensorflow.python.framework import ops
from os import path

module = tf.load_op_library(path.join(path.dirname(path.abspath(__file__)), 'lattice_filter.so'))


@ops.RegisterGradient("LatticeFilter")
def _lattice_filter_grad(op, grad):
    """ Gradients for the LatticeFilter op. We only need to calculate the gradients
    w.r.t. the first input (unaries) as we never need to backprop errors to the
    second input (reference image).

    Args:
    op: The `lattice_filter` operation that we are differentiating.
    grad: Gradients with respect to the output of the `lattice_filter` op.

    Returns:
    Gradients with respect to the input of `lattice_filter`.
    """

    reference_image = op.inputs[1]
    grad_vals = module.lattice_filter(grad, reference_image,
                                      bilateral=op.get_attr('bilateral'),
                                      theta_alpha=op.get_attr('theta_alpha'),
                                      theta_beta=op.get_attr('theta_beta'),
                                      theta_gamma=op.get_attr('theta_gamma'),
                                      reverse=True)

    return [grad_vals, tf.zeros_like(reference_image)]
