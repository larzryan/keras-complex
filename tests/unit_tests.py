import unittest

from tensorflow.keras.layers import Input, MaxPooling2D, Dense
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

import numpy as np
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

    def test_outputs_dense(self):
        """Test computed shape of dense layer output"""
        layer = conn.ComplexDense(units=16, activation='relu')
        input_shape = (None, 8)
        true = (None, 16 * 2)
        calc = layer.compute_output_shape(input_shape)
        self.assertEqual(true, calc)

    def test_dense_forward(self):
        """Test shape of model output, forward"""
        inputs = Input(shape=(128,))
        outputs = conn.ComplexDense(units=64, activation='relu')(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        true = (None, 64*2)
        calc = model.output_shape
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

    def test_github_example(self):
        # example from repository https://github.com/JesperDramsch/keras-complex/blob/master/README.md page
        model = tf.keras.models.Sequential()
        model.add(conn.conv.ComplexConv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 2)))
        model.add(conn.bn.ComplexBatchNormalization())
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        model.summary()

    def test_train_dense(self):
        inputs = 28
        outputs = 128
        # build a sequential complex dense model
        model = Sequential(name='complex')
        model.add(conn.ComplexDense(32, activation='relu', input_shape=(inputs*2,)))
        model.add(conn.ComplexBN())
        model.add(conn.ComplexDense(64, activation='relu'))
        model.add(conn.ComplexBN())
        model.add(Dense(128, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        # create some random data
        re = np.random.randn(inputs)
        im = np.random.randn(inputs)
        X = np.expand_dims(np.concatenate((re, im), -1), 0)
        Y = np.expand_dims(np.random.randn(outputs), 0)
        model.fit(X, Y, batch_size=1, epochs=10)


