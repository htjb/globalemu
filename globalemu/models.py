import tensorflow as tf


class network_models():
    def __init__(self):
        self.Model = tf.keras.models.Model
        self.Inputs = tf.keras.layers.Input
        self.Dense = tf.keras.layers.Dense
        self.Dropout = tf.keras.layers.Dropout

    def basic_model(
            self, input_dim, output_dim, layer_sizes, activation,
            drop_val, output_activation):
        a0 = self.Inputs(shape=(input_dim,))
        inputs = a0
        for layer_size in layer_sizes:
            outputs = self.Dense(layer_size, activation=activation)(a0)
            outputs = self.Dropout(drop_val)(outputs)
            a0 = outputs
        outputs = self.Dense(output_dim, activation=output_activation)(a0)
        model = self.Model(inputs, outputs)
        return model
