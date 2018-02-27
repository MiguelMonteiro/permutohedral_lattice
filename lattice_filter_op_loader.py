import tensorflow as tf
from tensorflow.python.framework import ops
module = tf.load_op_library('./lattice_filter.so')


@ops.RegisterGradient("LatticeFilter")
def _lattice_filter_grad(op, grad):
    """ Gradients for the HighDimFilter op. We only need to calculate the gradients
    w.r.t. the first input (unaries) as we never need to backprop errors to the
    second input (RGB values of the image).

    Args:
    op: The `high_dim_filter` operation that we are differentiating.
    grad: Gradients with respect to the output of the `high_dim_filter` op.

    Returns:
    Gradients with respect to the input of `high_dim_filter`.
    """

    reference_image = op.inputs[1]
    grad_vals = module.bilateral(grad, reference_image,
                                 bilateral=op.get_attr('bilateral'),
                                 theta_alpha=op.get_attr('theta_alpha'),
                                 theta_beta=op.get_attr('theta_beta'),
                                 theta_gamma=op.get_attr('theta_gamma'),
                                 reverse=True)

    return [grad_vals, tf.zeros_like(reference_image)]