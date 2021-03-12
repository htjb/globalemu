import numpy as np
from globalemu.downloads import download


class calc_signal:
    def __init__(self, z, base_dir):

        self.z = z
        self.base_dir = base_dir
        self.deltaT, self.T_K, self.T_s, self.T_r = self.calc()

    def calc(self):

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

        T_r = T_cmb0*(1+self.z)
        z_ref = 40
        T_K_ref = 33.7340  # K

        T_K = T_K_ref*((1+self.z)/(1+z_ref))**2

        Y = 0.274  # Helium abundance by mass
        rhoc = 1.36e11*(h/0.7)**2  # M_sol/cMpc^3
        mp = 8.40969762e-58  # m_p in M_sol
        nH = (rhoc/mp)*(1-Y)*omega_b*(1+self.z)**3*3.40368e-68

        Tstar = 0.068  # K
        try:
            t, kappa10_HH_data = np.loadtxt('kappa_HH.txt', unpack=True)
        except FileNotFoundError:
            download().kappa()
            t, kappa10_HH_data = np.loadtxt('kappa_HH.txt', unpack=True)

        kappa10_HH = np.interp(T_K, t, kappa10_HH_data)

        xc = (nH*kappa10_HH*1e-6*Tstar)/(A10*T_r)
        invT_s = (1/T_r + xc*(1/T_K))/(1+xc)
        T_s = 1/invT_s

        xHI = 1
        nu0 = 1420.4e6

        Hz = (H0)*np.sqrt(omega_m*(1+self.z)**3)

        tau = (3*planck_h*c**3*A10*xHI*nH) / \
            (32*np.pi*kb*T_s*nu0**2*(1+self.z)*Hz/(1+self.z))

        deltaT = (T_s-T_r)/(1+self.z)*(1-np.exp(-tau))

        norm_factor = np.load(self.base_dir + 'AFB_norm_factor.npy')
        deltaT = deltaT/np.abs(deltaT).max()*np.abs(norm_factor)*1e3
        np.savetxt(self.base_dir + 'AFB.txt', deltaT)

        return deltaT, T_K, T_s, T_r
