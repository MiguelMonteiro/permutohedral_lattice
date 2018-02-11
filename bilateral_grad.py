import tensorflow as tf
from tensorflow.python.framework import ops
module = tf.load_op_library('./bilateral.so')


@ops.RegisterGradient("Bilateral")
def _bilateral_grad(op, grad):
    """ Gradients for the HighDimFilter op. We only need to calculate the gradients
    w.r.t. the first input (unaries) as we never need to backprop errors to the
    second input (RGB values of the image).

    Args:
    op: The `high_dim_filter` operation that we are differentiating.
    grad: Gradients with respect to the output of the `high_dim_filter` op.

    Returns:
    Gradients with respect to the input of `high_dim_filter`.
    """

    feats = op.inputs[1]
    grad_vals = module.bilateral(grad, feats, reverse=True)

    return [grad_vals, tf.zeros_like(feats)]