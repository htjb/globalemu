"""
This function can be used to generate a configurate file for the GUI that
is specific to a given trained model. The file gets saved into the supplied
``base_dir`` which should contain the relevant trained model. The user
also needs to supply a path to the ``data_dir`` that contains the relevant
testing and training data. Additional arguments are described below.

A GUI config file is required to be able to visualise the signals with the
GUI and once generated the gui can be run from the command line

.. code:: bash

    globalemu /path/to/base_dir/containing/model/and/config/

"""

import numpy as np
import pickle


class config():

    r"""

    **Parameters:**

        base_dir: **string**
            | The path to the file containing the trained tensorflow model
                that the user wishes to visualise with the GUI. Must end
                in '/'.

        paramnames: **list of strings**
            | This should be a list of parameter names in the correct input
                order. For example for the released global signal model this
                would correspond to

                .. code: python

                    paramnames = [r'$\log(f_*)$', r'$\log(V_c)$',
                                  r'$\log(f_X)$',
                                  r'$\tau$', r'$\alpha$',
                                  r'$\nu_\mathrm{min}$',
                                  r'$R_\mathrm{mfp}$']

                Latex strings can be provided as above.

        data_dir: **string**
            | The file path to the training and test data which is used to set
                the y lims of the GUI graph and ranges/intervals of GUI
                sliders.

    **Kwargs:**

        logs: **list / default: [0, 1, 2]**
            | The indices corresponding to the astrophysical
                parameters that
                were logged during training. The default assumes
                that the first three columns in "train_data.txt" are
                :math:`{f_*}` (star formation efficiency),
                :math:`{V_c}` (minimum virial circular velocity) and
                :math:`{f_x}` (X-ray efficieny).

        ylabel: **string / default: 'y'**
            | y-axis label for gui plot.

    """

    def __init__(self, base_dir, paramnames, data_dir, **kwargs):

        for key, values in kwargs.items():
            if key not in set(
                    ['logs', 'ylabel']):
                raise KeyError("Unexpected keyword argument in process()")

        self.base_dir = base_dir
        self.paramnames = paramnames
        self.data_dir = data_dir
        self.logs = kwargs.pop('logs', [0, 1, 2])
        self.ylabel = kwargs.pop('ylabel', 'y')

        file_kwargs = [self.base_dir, self.data_dir]
        file_strings = ['base_dir', 'data_dir']
        for i in range(len(file_kwargs)):
            if type(file_kwargs[i]) is not str:
                raise TypeError("'" + file_strings[i] + "' must be a sting.")
            elif file_kwargs[i].endswith('/') is False:
                raise KeyError("'" + file_strings[i] + "' must end with '/'.")

        file = open(self.base_dir + "preprocess_settings.pkl", "rb")
        self.preprocess_settings = pickle.load(file)

        if type(self.paramnames) is not list:
            raise TypeError("'paramnames' must be a list of strings.")

        if type(self.logs) is not list:
            raise TypeError("'logs' must be a list.")

        test_data = np.loadtxt(data_dir + 'test_data.txt')
        test_labels = np.loadtxt(data_dir + 'test_labels.txt')
        for i in range(test_data.shape[1]):
            if i in self.logs:
                for j in range(test_data.shape[0]):
                    if test_data[j, i] == 0:
                        test_data[j, i] = 1e-6
                test_data[:, i] = np.log10(test_data[:, i])

        data_mins = test_data.min(axis=0)
        data_maxs = test_data.max(axis=0)

        full_logs = []
        for i in range(len(data_maxs)):
            if i in set(self.logs):
                full_logs.append(i)
            else:
                full_logs.append('--')

        # Create the data arrays
        n = len(data_maxs)
        names = np.array(self.paramnames, dtype='U100')  # Unicode strings
        mins = np.array(data_mins, dtype=float)
        maxs = np.array(data_maxs, dtype=float)

        # label_min: first value is test_labels.min(), rest are empty strings
        label_min = np.array([test_labels.min()] + ['']*(n-1), dtype='U100')

        # label_max: first value is test_labels.max(), rest are empty strings
        label_max = np.array([test_labels.max()] + ['']*(n-1), dtype='U100')

        logs = np.array(full_logs, dtype='U100')
        ylabel = np.array([self.ylabel]*n, dtype='U100')

        # Stack into 2D array (transpose to get columns)
        data = np.column_stack([names, mins, maxs, label_min, label_max, logs, ylabel])

        # Write header
        header = 'names,mins,maxs,label_min,label_max,logs,ylabel'

        # Save to CSV
        np.savetxt(
            base_dir + 'gui_configuration.csv',
            data,
            delimiter=',',
            header=header,
            comments='',  # Prevents '#' being added to header
            fmt='%s'      # String format for all columns
        )
