
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op
import high_dim_filter2d_grad
custom_module = high_dim_filter2d_grad.custom_module

class HighDimGradTest(tf.test.TestCase):

    def test_high_dim_filter_grad(self):
        x_shape = [10, 10, 3]
        rgb_shape = [10, 10, 3]

        # CppTests inputs: unaries and RGB values
        unary_np = np.random.randn(*x_shape).astype(np.float32)
        rgb_np = np.random.randint(low=0, high=256, size=rgb_shape).astype(np.float32) #/ 256.0

        with self.test_session():
            unary_tf = constant_op.constant(unary_np)
            rgb_tf = constant_op.constant(rgb_np)
            y_tf = custom_module.high_dim_filter2d(unary_tf, rgb_tf, reverse=True)

            out = gradient_checker.compute_gradient([unary_tf, rgb_tf], [x_shape, rgb_shape], y_tf, x_shape)

            # We only need to compare gradients w.r.t. unaries
            computed = out[0][0].flatten()
            estimated = out[0][1].flatten()
            np.savetxt('computed.txt',computed,'%.6f')
            np.savetxt('estimated.txt',estimated,'%.6f')

            mask = (computed != 0)
            computed = computed[mask]
            estimated = estimated[mask]
            difference = computed - estimated

            measure1 = np.mean(difference) / np.mean(computed)
            measure2 = np.max(difference) / np.max(computed)

            print('Gradient check: measure1 = {:.6f}, measure2 = {:.6f}'.format(measure1, measure2))
            self.assertLess(measure1, 1e-3, 'Errors found in the gradient computation.')
            self.assertLess(measure2, 2e-2, 'Errors found in the gradient computation.')
            print('Gradient check: success!')


if __name__ == '__main__':
    tf.test.main()
