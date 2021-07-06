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
import pickle


class evaluate():

    r"""

    The class can be initialised with the following kwargs and the
    following code

    .. code:: python

        predictor = evaluate(**kwargs)

    **kwargs:**

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
                You can pass a single set of parameters or a 2D array of
                different parameters to evaluate. For example if I wanted to
                evaluate 100 sets of 7 parameters my input array should have
                shape=(100, 7).

    **Return:**

        signal: **array or float**
            | The emulated signal. If a single redshift is passed to the
                emulator then the returned signal will be a single float
                otherwise the result will be an array. If more than
                one set of parameters are input then the output signal will
                be an array of signals. e.g. 100 input sets of parameters
                gives signal.shape=(100, len(z)).

        z: **array or float**
            | The redshift values corresponding to the returned signal. If
                z was not specified on input then the returned signal and
                redshifts will correspond to the redshifts that the network
                was originally trained on.

    """

    def __init__(self, **kwargs):

        for key, values in kwargs.items():
            if key not in set(
                    ['base_dir', 'model', 'logs', 'gc', 'z']):
                raise KeyError("Unexpected keyword argument in evaluate()")

        self.base_dir = kwargs.pop('base_dir', 'model_dir/')
        if type(self.base_dir) is not str:
            raise TypeError("'base_dir' must be a sting.")
        elif self.base_dir.endswith('/') is False:
            raise KeyError("'base_dir' must end with '/'.")

        file = open(self.base_dir + "preprocess_settings.pkl", "rb")
        self.preprocess_settings = pickle.load(file)

        self.data_mins = np.loadtxt(self.base_dir + 'data_mins.txt')
        self.data_maxs = np.loadtxt(self.base_dir + 'data_maxs.txt')
        if self.preprocess_settings['resampling'] is True:
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

        boolean_kwargs = [self.garbage_collection]
        boolean_strings = ['gc']
        for i in range(len(boolean_kwargs)):
            if type(boolean_kwargs[i]) is not bool:
                raise TypeError("'" + boolean_strings[i] + "' must be a bool.")

        if self.preprocess_settings['AFB'] is True:
            self.AFB = np.loadtxt(self.base_dir + 'AFB.txt')
        if self.preprocess_settings['std_division'] is True:
            self.label_stds = np.load(self.base_dir + 'labels_stds.npy')

        self.original_z = np.loadtxt(self.base_dir + 'z.txt')

        self.z = kwargs.pop('z', self.original_z)
        if type(self.z) not in set([np.ndarray, list, int, float]):
            raise TypeError("'z' should be a numpy array, list, float or int.")

    def __call__(self, parameters):

        if type(parameters) not in set([np.ndarray, list]):
            raise TypeError("'params' must be a list or np.array.")
        if type(parameters) is list:
            parameters = np.array(parameters)

        if len(parameters.shape) == 1:
            params = []
            for i in range(len(parameters)):
                if i in set(self.logs):
                    if parameters[i] == 0:
                        parameters[i] = 1e-6
                    params.append(np.log10(parameters[i]))
                else:
                    params.append(parameters[i])
            normalised_params = np.array([
                (params[i] - self.data_mins[i]) /
                (self.data_maxs[i] - self.data_mins[i])
                for i in range(len(params))])
        else:
            params = []
            for i in range(len(parameters)):
                params_set = []
                for j in range(len(parameters[i])):
                    if j in set(self.logs):
                        if parameters[i, j] == 0:
                            parameters[i, j] = 1e-6
                        params_set.append(np.log10(parameters[i, j]))
                    else:
                        params_set.append(parameters[i, j])
                params.append(params_set)
            params = np.array(params)
            normalised_params = np.array([
                (params[:, i] - self.data_mins[i]) /
                (self.data_maxs[i] - self.data_mins[i])
                for i in range(params.shape[1])]).T

        if self.preprocess_settings['resampling'] is True:
            norm_z = np.interp(self.z, self.original_z, self.cdf)
        else:
            norm_z = (self.z - self.original_z.min()) / \
                     (self.original_z.max() - self.original_z.min())
        if isinstance(norm_z, np.ndarray):
            if len(normalised_params.shape) == 1:
                x = np.tile(normalised_params, (len(norm_z), 1))
                x = np.hstack([x, norm_z[:, np.newaxis]])
            else:
                x = np.hstack([
                               np.vstack([np.tile(normalised_params[i],
                                         (len(norm_z), 1))
                                         for i in
                                         range(len(normalised_params))]),
                               np.vstack([norm_z[:, np.newaxis]] *
                                         normalised_params.shape[0])])
            tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            result = self.model(tensor, training=False).numpy()
            if len(normalised_params.shape) != 1:
                evaluation = np.array([
                    result[i:i + len(self.z)]
                    for i in range(0, len(result), len(self.z))])[:, :, 0]
            else:
                evaluation = result.T[0]
            if self.garbage_collection is True:
                K.clear_session()
                gc.collect()
        else:
            x = np.hstack([normalised_params, norm_z]).astype(np.float32)
            result = self.model(x[np.newaxis, :], training=False).numpy()
            evaluation = result[0][0]

        if self.preprocess_settings['std_division'] is True:
            if isinstance(evaluation, np.ndarray):
                evaluation = [
                    evaluation[i]*self.label_stds
                    for i in range(evaluation.shape[0])]
            else:
                evaluation *= self.label_stds

        if self.preprocess_settings['AFB'] is True:
            evaluation += np.interp(self.z, self.original_z, self.AFB)

        if type(evaluation) is not np.ndarray:
            evaluation = np.array(evaluation)
        
        return evaluation, self.z
