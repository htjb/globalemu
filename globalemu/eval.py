"""

``evaluate()`` is used to make an evaluation of a trained instance of
``globalemu``. It has to be initialised with a set of kwargs, most importantly
the ``base_dir`` which contains the trained model. Once initialised it can
then be used to make predictions and return the predicted signal plus the
corresponding redshift. ``evaluate()`` can reproduce a high resolution Global
21-cm signal (450 redshift data points) in 1.5 ms.

"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import gc


class evaluate():

    r"""

    The class can be initialised with the following kwargs and the
    following code

    .. code:: python

        predictor = evaluate(**kwargs)

    **kwargs:**

        xHI: **Bool / default: False**
            | If True then ``globalemu`` will act as if it is evaluating a
                neutral fraction history emulator.

        base_dir: **string / default: 'model_dir/'**
            | The ``base_dir`` is where the trained model is saved.

        model: **tensorflow model / default: None**
            | If making multiple calls to the function it is advisable to
                load the trained model in the script making the calls and
                then to pass it to ``evaluate()``. This prevents the model
                being loaded upon each call and leads to a significant
                increase speed. You can load a model via,

                .. code:: python

                    from tensorflow import keras

                    model = keras.models.load_model(
                        base_dir + 'model.h5',
                        compile=False)

        logs: **list / default: [0, 1, 2]**
            | The indices corresponding to the astrophysical
                parameters that
                were logged during training. The default assumes
                that the first three columns in "train_data.txt" are
                :math:`{f_*}` (star formation efficiency),
                :math:`{V_c}` (minimum virial circular velocity) and
                :math:`{f_x}` (X-ray efficieny).

        gc: **Bool / default: False**
            | Multiple calls to the function can cause runaway memory
                related issues (it is worth testing this
                behaviour before scheduling
                hpc jobs) and these memory issues can be somewhat eleviated
                by setting ``gc=True``. This performs a garbage collection
                after every function call. It is an optional argumen set to
                ``False`` by default because it can increase the time taken
                to perform the emulation.

        z: **list or np.array / default: Original redshift array**
            | The redshift values at which you want to emulate the 21-cm
                signal. The default is given by the redshift range that the
                network was originally trained on (found in ``base_dir``).

    Once the class has been initialised you can then make evaluations
    of the emulator by passing the parameters like so

    .. code:: python

        signal, z = predictor(parameters)

    **Parameters:**

        parameters: **list or np.array**
            | The combination of astrophysical parameters that you want to
                emulate a global signal for. They must be in the same order
                as was used when training and they must fall with in the
                trained parameter space. For the 21cmGEM data the order
                of the astrophysical parameters is given by:
                :math:`{f_*, V_c, f_x, \tau, \alpha, \nu_\mathrm{min}}` and
                :math:`{R_\mathrm{mfp}}` (see the ``globalemu`` paper and
                references therein for a description of the parameters).

    **Return:**

        signal: **array or float**
            | The emulated signal. If a single redshift is passed to the
                emulator then the returned signal will be a single float
                otherwise the result will be an array.

        z: **array or float**
            | The redshift values corresponding to the returned signal. If
                z was not specified on input then the returned signal and
                redshifts will correspond to the redshifts that the network
                was originally trained on.

    """

    def __init__(self, **kwargs):

        for key, values in kwargs.items():
            if key not in set(
                    ['xHI', 'base_dir', 'model', 'logs', 'gc', 'z']):
                raise KeyError("Unexpected keyword argument in evaluate()")

        self.xHI = kwargs.pop('xHI', False)

        self.base_dir = kwargs.pop('base_dir', 'model_dir/')
        if type(self.base_dir) is not str:
            raise TypeError("'base_dir' must be a sting.")
        elif self.base_dir.endswith('/') is False:
            raise KeyError("'base_dir' must end with '/'.")

        self.data_mins = np.loadtxt(self.base_dir + 'data_mins.txt')
        self.data_maxs = np.loadtxt(self.base_dir + 'data_maxs.txt')
        self.cdf = np.loadtxt(self.base_dir + 'cdf.txt')

        self.model = kwargs.pop('model', None)
        if self.model is None:
            self.model = keras.models.load_model(
                self.base_dir + 'model.h5',
                compile=False)

        self.logs = kwargs.pop('logs', [0, 1, 2])
        if type(self.logs) is not list:
            raise TypeError("'logs' must be a list.")
        self.garbage_collection = kwargs.pop('gc', False)

        boolean_kwargs = [self.garbage_collection, self.xHI]
        boolean_strings = ['gc', 'xHI']
        for i in range(len(boolean_kwargs)):
            if type(boolean_kwargs[i]) is not bool:
                raise TypeError("'" + boolean_strings[i] + "' must be a bool.")

        if self.xHI is False:
            self.AFB = np.loadtxt(self.base_dir + 'AFB.txt')
            self.label_stds = np.load(self.base_dir + 'labels_stds.npy')

        self.original_z = np.loadtxt(self.base_dir + 'z.txt')

        self.z = kwargs.pop('z', self.original_z)
        if type(self.z) not in set([np.ndarray, list, int, float]):
            raise TypeError("'z' should be a numpy array, list, float or int.")

    def __call__(self, parameters):

        if type(parameters) not in set([np.ndarray, list]):
            raise TypeError("'params' must be a list or np.array.")

        params = []
        for i in range(len(parameters)):
            if i in set(self.logs):
                if parameters[i] == 0:
                    parameters[i] = 1e-6
                params.append(np.log10(parameters[i]))
            else:
                params.append(parameters[i])

        normalised_params = [
            (params[i] - self.data_mins[i]) /
            (self.data_maxs[i] - self.data_mins[i])
            for i in range(len(params))]

        norm_z = np.interp(self.z, self.original_z, self.cdf)

        if isinstance(norm_z, np.ndarray):
            x = np.tile(normalised_params, (len(norm_z), 1))
            x = np.hstack([x, norm_z[:, np.newaxis]])
            tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            result = self.model(tensor, training=False).numpy()
            evaluation = result.T[0]
            if self.garbage_collection is True:
                K.clear_session()
                gc.collect()
        else:
            x = np.hstack([normalised_params, norm_z]).astype(np.float32)
            result = self.model(x[np.newaxis, :], training=False).numpy()
            evaluation = result[0][0]

        if self.xHI is False:
            if isinstance(evaluation, np.ndarray):
                evaluation = [
                    evaluation[i]*self.label_stds
                    for i in range(evaluation.shape[0])]
            else:
                evaluation *= self.label_stds

            evaluation += np.interp(self.z, self.original_z, self.AFB)

        return evaluation, self.z
