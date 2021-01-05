import numpy as np
import matplotlib.pyplot as plt

class calc_signal:
    def __init__(self, z, **kwargs):
        self.A10 = 2.85e-15 #s^-1
        self.kb = 1.381e-23 #m^2 kg s^-2 K^-1

        # Furlanetto pg 13
        #self.h = 0.74
        #self.H0 = 74 * 1000/3.089e22 # s^-1
        #self.omega_b = 0.044
        #self.omega_m = 0.26
        #self.omega_lam = 0.74

        # 21cmGEM code
        self.h = 0.6704
        self.H0 = 100*self.h * 1000/3.089e22
        self.omega_c = 0.12038/self.h**2
        self.omega_b = 0.022032/self.h**2
        self.omega_m = self.omega_b + self.omega_c
        #print(self.omega_m)
        self.omega_lam = 1 - self.omega_m

        self.T_cmb0 = 2.725 # K
        self.z = z
        self.orig_z = np.linspace(5, 50, 451)
        self.planck_h = 6.626e-34 # m^2 kg s^-1
        self.c = 3e8 # m/s

        self.collisions = kwargs.pop('collisions', True)
        self.base_dir = kwargs.pop('base_dir', 'results/')

        self.deltaT, self.T_K, self.T_s, self.T_r = self.calc()

    def calc(self):

        T_r = self.T_cmb0*(1+self.orig_z)
        #z_ref = 50
        #T_K_ref = 50.6660 #K
        z_ref = 40
        T_K_ref = 33.7340#K

        T_K = T_K_ref*((1+self.orig_z)/(1+z_ref))**2

        Y = 0.274 #Helium abundance by mass
        rhoc = 1.36e11*(self.h/0.7)**2 #M_sol/cMpc^3
        mp = 8.40969762e-58 # m_p in M_sol
        nH = (rhoc/mp)*(1-Y)*self.omega_b*(1+self.orig_z)**3*3.40368e-68

        if self.collisions is False:
            T_s = T_K.copy()
        if self.collisions is True:
            Tstar = 0.068 #K
            t, kappa10_HH_data = np.loadtxt('kappa_HH.txt', unpack=True)
            kappa10_HH = np.interp(T_K, t, kappa10_HH_data)

            xc = (nH*kappa10_HH*1e-6*Tstar)/(self.A10*T_r)
            invT_s = (1/T_r + xc*(1/T_K))/(1+xc)
            T_s = 1/invT_s

        xHI = 1
        nu0 = 1420.4e6

        Hz = (self.H0)*np.sqrt(self.omega_m*(1+self.orig_z)**3)#+self.omega_lam)

        tau = (3*self.planck_h*self.c**3*self.A10*xHI*nH)/ \
            (32*np.pi*self.kb*T_s*nu0**2*(1+self.orig_z)*Hz/(1+self.orig_z))

        deltaT = (T_s-T_r)/(1+self.orig_z)*(1-np.exp(-tau))

        norm_factor = np.load(self.base_dir + 'AFB_norm_factor.npy')
        deltaT = deltaT/np.abs(deltaT).max()*np.abs(norm_factor)*1e3

        deltaT = np.interp(self.z, self.orig_z, deltaT)
        T_K = np.interp(self.z, self.orig_z, T_K)
        T_s = np.interp(self.z, self.orig_z, T_s)
        T_r = np.interp(self.z, self.orig_z, T_r)

        return deltaT, T_K, T_s, T_r
