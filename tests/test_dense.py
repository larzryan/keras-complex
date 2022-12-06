import unittest

from tensorflow.keras.layers import Input, MaxPooling2D, Dense
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

import numpy as np
import complexnn as conn


class TestDenseMethods(unittest.TestCase):
    """Test Dense layer"""

    def test_outputs_dense(self):
        """Test computed shape of dense layer output"""
        layer = conn.ComplexDense(units=16, activation="relu")
        input_shape = (None, 8)
        true = (None, 16 * 2)
        calc = layer.compute_output_shape(input_shape)
        self.assertEqual(true, calc)

    def test_outputs_dense(self):
        """Test computed shape of dense layer output"""
        layer = conn.ComplexDense(units=16, activation="relu")
        input_shape = (None, 8)
        true = (None, 16 * 2)
        calc = layer.compute_output_shape(input_shape)
        self.assertEqual(true, calc)
