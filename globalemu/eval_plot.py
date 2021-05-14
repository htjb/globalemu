"""

This function can be used to assess the accuracy of emulation of a test data
set given a trained model and produces a figure showing the
mean, 95th percentile and worst emulations. Examples of these figures can be
found in the `MNRAS preprint <https://arxiv.org/abs/2104.04336)>`__.

"""

import numpy as np
from globalemu.eval import evaluate
from globalemu.losses import loss_functions
import matplotlib.pyplot as plt

class signal_plot():

    r"""

    The class can be initialised with the following kwargs and the
    following code

    .. code:: python

        plotter  = signal_plot(parameters, labels, loss_type,
                        predictor, **kwargs)

    **Parameters:**

        parameters: **list or np.array**
            | The astrophysical parameters corresponding to the testing data.

        labels: **list or np.array**
            | The signals, corresponding to the input parameters, that
                we want to predict and subsequently plot the mean, 95th
                percentile and worst emulations of.

        loss_type: ** str or function**
            | The metric by which we want to assess the accuracy of emulation.
                The built in loss functions can be accessed by setting this
                variable to 'rmse', 'mse' or 'GEMLoss'. Alternatively, a user
                defined callable function that takes in the labels and signals
                can also be provided.

        predictor: ** globalemu.eval object **
            | An instance of the globalemu eval class that will be used to make
                predictions of the labels from the input parameters.

    **kwargs:**

        rtol: **int or float / default: 1e-2**
            | The relative accuracy with which the function finds a
                signal with a loss equal to the mean loss for all predictions.

        atol: **int or float / default: 1e-2**
            | The absolute accuracy with which the function finds a
                signal with a loss equal to the mean loss for all predictions.

        figsizex: **int or float / default: 5**
            | The of the figure along the x axis to be passed to
                plt.subplots().

        figsizey: **int or float / default: 10**
            | The of the figure along the y axis to be passed to
                plt.subplots().

    Once the class has been initialised you can then call the plotter like so

    .. code:: python

        fig, axes = plotter()

    **Return:**

    """

    def __init__(self, parameters, labels, loss_type,
                    predictor, **kwargs):

        self.rtol = kwargs.pop('rtol', 1e-2)
        self.atol = kwargs.pop('atol', 1e-2)
        self.figsizex = kwargs.pop('figsizex', 5)
        self.figsizey = kwargs.pop('figsizey', 10)

        float_kwargs = [self.rtol, self.atol, self.figsizex, self.figsizey]
        float_kwarg_str = ['rtol', 'atol', 'figsizex', 'figsizey']
        for i in range(float_kwargs):
            if type(float_kwargs[i]) not in set([float, int]):
                raise TypeError("'" + float_kwarg_str[i] +
                                "' must be an integer or a float.")

        self.parameters = parameters

        if type(self.parameters) not in set([np.ndarray, list]):
            raise TypeError("'parameters' must be a list or np.array.")

        self.labels = labels

        if type(self.labels) not in set([np.ndarray, list]):
            raise TypeError("'labels' must be a list or np.array.")

        self.loss_type = loss_type

        if type(loss_type) not str:
            if not callable(loss_type):
                raise TypeError("'loss_type' must be a string from the " +
                                "predefined set (see documentaiton) or a " +
                                "user defined function.")

        self.predictor = predictor

        if not callable(predictor):
            raise TypeError("'predictor' should be an instance of " +
                            "globalemu.eval.")

    def __call__(self):

        signal, z = self.predictor(self.parameters)

        loss = []
        for i in range(len(signal)):
            if type(self.loss_type) is not str:
                loss.append(loss_type(self.labels[i], signal[i]))
            else:
                lf = loss_functions(self.labels[i], signal[i])
                if self.loss_type == 'rmse':
                    loss.append(lf.rmse().numpy())
                elif self.loss_type == 'mse':
                    loss.append(lf.mse())
                elif self.loss_type == 'GEMLoss':
                    loss.append(lf.GEMLoss())
        loss = np.array(loss)

        mean_label = self.labels[
                    np.where(
                             np.isclose(
                                        loss, loss.mean(),
                                        rtol=self.rtol,
                                        atol=self.atol))[0][0], :]
        mean_pred = signal[
                    np.where(
                             np.isclose(
                                        loss, loss.mean(),
                                        rtol=self.rtol,
                                        atol=self.atol))[0][0], :]
        worst_label = self.labels[np.where(loss == loss.max())[0][0], :]
        worst_pred = signal[np.where(loss == loss.max())[0][0], :]

        args = np.argsort(loss)
        sorted_loss = loss[args]
        sorted_labels = self.labels[args]
        sorted_signals = signal[args]
        idx = int(len(sorted_loss)/100*95)
        limit95 = sorted_loss[idx]
        limit_label = sorted_labels[idx, :]
        limit_pred = sorted_signals[idx, :]

        fig, axes = plt.subplots(3, 1,
                                 figsize=(self.figsizex, self.figsizey),
                                 sharex=True)
        axes[0].plot(z, mean_label, label='True Signal')
        axes[0].plot(z, mean_pred, label= 'Loss = {:.3f}'.format(
            loss[
                 np.where(np.isclose(loss, loss.mean(),
                          rtol=self.rtol, atol=self.atol))[0][0]]))
        axes[1].plot(z, limit_label, label='True Signal')
        axes[1].plot(z, limit_pred, label= 'Loss = {:.3f}'.format(limit95))
        axes[2].plot(z, worst_label, label='True Signal')
        axes[2].plot(z, worst_pred, label= 'Loss = {:.3f}'.format(loss.max()))
        axes[0].legend(title='Mean:')
        axes[1].legend(title='95%:')
        axes[2].legend(title='Worst:')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)

        return fig, axes
