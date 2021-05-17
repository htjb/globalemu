import numpy as np
from globalemu.downloads import download


class calc_signal():

    r"""

    The code calculates the astrophysics free baseline (AFB) for the
    global 21-cm
    signal during the training of the emulator. For details on the
    mathematics see the globalemu MNRAS preprint at
    https://arxiv.org/abs/2104.04336.

    The AFB is saved in the base_dir for later use when evaluating the
    network.

    **Parameters:**

        z: **list or np.array**
            | The array of redshift values to calculate the AFB at.

        base_dir: **string**
            | The location of the trained model.
    """

    def __init__(self, z, base_dir):

        self.z = z
        self.base_dir = base_dir

        A10 = 2.85e-15  # s^-1
        kb = 1.381e-23  # m^2 kg s^-2 K^-1

        # 21cmGEM code
        h = 0.6704
        H0 = 100*h * 1000/3.089e22
        omega_c = 0.12038/h**2
        omega_b = 0.022032/h**2
        omega_m = omega_b + omega_c

        T_cmb0 = 2.725  # K
        planck_h = 6.626e-34  # m^2 kg s^-1
        c = 3e8  # m/s

        self.T_r = T_cmb0*(1+self.z)
        z_ref = 40
        T_K_ref = 33.7340  # K

        self.T_K = T_K_ref*((1+self.z)/(1+z_ref))**2

        Y = 0.274  # Helium abundance by mass
        rhoc = 1.36e11*(h/0.7)**2  # M_sol/cMpc^3
        mp = 8.40969762e-58  # m_p in M_sol
        nH = (rhoc/mp)*(1-Y)*omega_b*(1+self.z)**3*3.40368e-68

        Tstar = 0.068  # K
        try:
            t, kappa10_HH_data = np.loadtxt('kappa_HH.txt', unpack=True)
        except OSError:
            download().kappa()
            t, kappa10_HH_data = np.loadtxt('kappa_HH.txt', unpack=True)

        kappa10_HH = np.interp(self.T_K, t, kappa10_HH_data)

        xc = (nH*kappa10_HH*1e-6*Tstar)/(A10*self.T_r)
        invT_s = (1/self.T_r + xc*(1/self.T_K))/(1+xc)
        self.T_s = 1/invT_s

        xHI = 1
        nu0 = 1420.4e6

        Hz = H0*np.sqrt(omega_m*(1+self.z)**3)

        tau = (3*planck_h*c**3*A10*xHI*nH) / \
            (32*np.pi*kb*self.T_s*nu0**2*(1+self.z)*Hz/(1+self.z))

        deltaT = (self.T_s-self.T_r)/(1+self.z)*(1-np.exp(-tau))

        norm_factor = np.load(self.base_dir + 'AFB_norm_factor.npy')
        self.deltaT = deltaT/np.abs(deltaT).max()*np.abs(norm_factor)*1e3
        np.savetxt(self.base_dir + 'AFB.txt', self.deltaT)
