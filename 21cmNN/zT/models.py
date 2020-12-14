import numpy as np
import tensorflow as tf

class network_models():
    def __init__(self):
        self.Model = tf.keras.models.Model
        self.Inputs = tf.keras.layers.Input
        self.Dense = tf.keras.layers.Dense
        self.Dropout = tf.keras.layers.Dropout
        self.BN = tf.keras.layers.BatchNormalization

    def basic_model(self, input_dim, output_dim, layer_sizes, activation, drop_val):
        a0 = self.Inputs(shape = (input_dim,))
        inputs = a0
        for layer_size in layer_sizes:
            outputs = self.Dense(layer_size, activation=activation)(a0)
            outputs = self.Dropout(drop_val)(outputs)
            a0 = outputs
        outputs = self.Dense(output_dim, activation='linear')(a0)
        model = self.Model(inputs, outputs)
        return model

    def basic_model_norm(self, input_dim, output_dim, layer_sizes, activation, drop_val):
        a0 = self.Inputs(shape = (input_dim,))
        inputs = a0
        for layer_size in layer_sizes:
            outputs = self.Dense(layer_size, activation=activation)(a0)
            outputs = self.Dropout(drop_val)(outputs)
            outputs = self.BN()(outputs)
            a0 = outputs
        outputs = self.Dense(output_dim, activation='linear')(a0)
        model = self.Model(inputs, outputs)
        return model

    def basic_model_L2(self, input_dim, output_dim, layer_sizes, activation, drop_val):
        a0 = self.Inputs(shape = (input_dim,))
        reg = tf.keras.regularizers.L2(l2=0.1)
        inputs = a0
        for layer_size in layer_sizes:
            outputs = self.Dense(layer_size, activation=activation, kernel_regularizer=reg)(a0)
            outputs = self.Dropout(drop_val)(outputs)
            a0 = outputs
        outputs = self.Dense(output_dim, activation='linear')(a0)
        model = self.Model(inputs, outputs)
        return model
