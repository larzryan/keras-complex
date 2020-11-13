import unittest

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf

import numpy as np
# import tensorflow as tf

import complexnn as conn


class TestDNCMethods(unittest.TestCase):
    """Unit test class"""

    def test_outputs_forward(self):
        """Test computed shape of forward convolution output"""
        layer = conn.ComplexConv2D(
            filters=4,
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=False)
        input_shape = (None, 128, 128, 2)
        true = (None, 64, 64, 8)
        calc = layer.compute_output_shape(input_shape)
        self.assertEqual(true, calc)

    def test_outputs_transpose(self):
        """Test computed shape of transposed convolution output"""
        layer = conn.ComplexConv2D(
            filters=2,
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=True)
        input_shape = (None, 64, 64, 4)
        true = (None, 128, 128, 4)
        calc = layer.compute_output_shape(input_shape)
        self.assertEqual(true, calc)

    def test_conv2Dforward(self):
        """Test shape of model output, forward"""
        inputs = Input(shape=(128, 128, 2))
        outputs = conn.ComplexConv2D(
            filters=4,
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=False)(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        true = (None, 64, 64, 8)
        calc = model.output_shape
        self.assertEqual(true, calc)

    def test_conv2Dtranspose(self):
        """Test shape of model output, transposed"""
        inputs = Input(shape=(64, 64, 20))  # = 10 CDN filters
        outputs = conn.ComplexConv2D(
            filters=2,  # = 4 Keras filters
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=True)(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        true = (None, 128, 128, 4)
        calc = model.output_shape
        self.assertEqual(true, calc)

    def test_train_transpose(self):
        """Train using Conv2DTranspose"""
        x = np.random.randn(64 * 64).reshape((64, 64))
        y = np.random.randn(64 * 64).reshape((64, 64))
        X = np.stack((x, y), -1)
        X = np.expand_dims(X, 0)
        Y = X
        inputs = Input(shape=(64, 64, 2))
        conv1 = conn.ComplexConv2D(
            filters=2,  # = 4 Keras filters
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=False)(inputs)
        outputs = conn.ComplexConv2D(
            filters=1,  # = 2 Keras filters => 1 complex layer
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=True)(conv1)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy'])
        model.fit(X, Y, batch_size=1, epochs=10)

    # def test_later():
        # tf.reset_default_graph()
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # tf_answer = np.array(sess.run())


