import tensorflow as tf


class network_models():

    r"""

    This function is used during training to generate the tensorflow model
    for training.

    """
    def __init__(self):
        self.Model = tf.keras.models.Model
        self.Inputs = tf.keras.layers.Input
        self.Dense = tf.keras.layers.Dense
        self.Dropout = tf.keras.layers.Dropout

    def basic_model(
            self, input_dim, output_dim, layer_sizes, activation,
            drop_val, output_activation):

        r"""

        The basic tensorflow model used in globalemu. The parameters are set
        in network.py.

        **Parameters:**

        input_dim: **int**
            | Number of input nodes for the network.

        output_dim: **int**
            | Number of output nodes for the network.

        layer_size: **list**
            | A list containing the structure of the hidden layers in the
                network. The number of elements in the list corresponds
                to the number of hidden layers and the value of each element
                to the number of nodes in that layer. e.g. [12, 12]
                corresponds to 2 hidden layers each with 12 nodes.

        """
        a0 = self.Inputs(shape=(input_dim,))
        inputs = a0
        for layer_size in layer_sizes:
            outputs = self.Dense(layer_size, activation=activation)(a0)
            outputs = self.Dropout(drop_val)(outputs)
            a0 = outputs
        outputs = self.Dense(output_dim, activation=output_activation)(a0)
        model = self.Model(inputs, outputs)
        return model
