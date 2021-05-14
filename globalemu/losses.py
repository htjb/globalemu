from tensorflow.keras import backend as K


class loss_functions():

    r"""

    The loss functions used to assess the quality of the emulation.

    **Parameters:**

    y: **tf.tensor**
        | The training labels for a given batch of training data.

    y_: **tf.tensor**
        | The predicted value of the training labels corresponding to y.

    """

    def __init__(self, y, y_):
        self.y = y
        self.y_ = y_

    def rmse(self):
        return K.sqrt(K.mean(K.square(self.y - self.y_)))

    def mse(self):
        return K.mean(K.square(self.y - self.y_))

    def GEMLoss(self):
        """
        This loss function corresponds to eq. 11 in the 21cmGEM paper
        (https://arxiv.org/abs/1910.06274) and
        is used in the globalemu MNRAS paper to compare the performance of the
        two emulators.
        """
        return (
            K.sqrt(K.mean(K.square(self.y - self.y_))) /
            K.max(K.abs(self.y)))
