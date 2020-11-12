import numpy as np
import matplotlib.pyplot as plt

class calc_signal:
    def __init__(self, z, **kwargs):
        self.A10 = 2.85e-15 #s^-1
        self.kb = 1.381e-23 #m^2 kg s^-2 K^-1

        # Furlanetto pg 13
        #self.h = 0.74
        #self.H0 = 74 * 1000/3.089e22 # s^-1
        self.omega_b = 0.044
        self.omega_m = 0.26
        self.omega_lam = 0.74

        # 21cmGEM code
        self.h = 0.6704
        self.H0 = 100*self.h * 1000/3.089e22

        self.T_cmb0 = 2.75 # K
        self.z = z
        self.orig_z = np.arange(5, 50.1, 0.1)
        self.planck_h = 6.626e-34 # m^2 kg s^-1
        self.c = 3e8 # m/s

        self.reionization = kwargs.pop('reionization', 'tanh')
        self.collisions = kwargs.pop('collisions', True)

        self.deltaT, self.T_K, self.T_s, self.T_r = self.calc()

    def calc(self):

        z_dec = 150*((self.omega_b*self.h**2)/0.023)**(2/5) - 1
        T_K_dec = self.T_cmb0*(1+z_dec)
        T_r = self.T_cmb0*(1+self.orig_z)

        T_K = np.empty(len(self.orig_z))
        for i in range(len(self.orig_z)):
            if self.orig_z[i] < z_dec:
                T_K[i] = T_K_dec*(1+self.orig_z[i])**2/(1+z_dec)**2
            else:
                T_K[i] = T_r[i]

        #The Intergalactic Medium, Piero Madau
        nH = 1.67e-7*(self.omega_b*self.h**2)/0.019*(1+self.orig_z)**3*1e6

        if self.collisions is False:
            T_s = T_K.copy()
        if self.collisions is True:
            Tstar = 0.068 #K
            t, kappa10_HH_data = np.loadtxt('kappa_HH.txt', unpack=True)
            kappa10_HH = np.interp(T_K, t, kappa10_HH_data)

            xc = (nH*kappa10_HH*1e-6*Tstar)/(self.A10*T_r)
            invT_s = (1/T_r + xc*(1/T_K))/(1+xc)
            T_s = 1/invT_s

        if self.reionization == 'unity':
            xHI = 1
        if self.reionization == 'tanh':
            # Witnessing the reionization history using CMB observations from
            # Planck, D. K. Hazra and G. F. Smoot
            Fe = 0.08
            dz = 0.5
            zre = 10
            y = (1 + self.orig_z)**(3/2)
            yre = (1 + zre)**(3/2)
            DeltaRegion = 1.5*np.sqrt(1+zre)*dz
            xe = ((1+Fe)/2)*(1 + np.tanh((yre - y)/DeltaRegion))
            xHI = 1 - xe

        nu0 = 1420.4e6

        Hz = (self.H0)*np.sqrt(self.omega_m*(1+self.orig_z)**3+self.omega_lam)

        tau = (3*self.planck_h*self.c**3*self.A10*xHI*nH)/ \
            (32*np.pi*self.kb*T_s*nu0**2*(1+self.orig_z)*Hz/(1+self.orig_z))

        deltaT = (T_s-T_r)/(1+self.orig_z)*(1-np.exp(-tau))

        #if np.all(self.z == self.orig_z):
        deltaT = np.interp(self.z, self.orig_z, deltaT)
        T_K = np.interp(self.z, self.orig_z, T_K)
        T_s = np.interp(self.z, self.orig_z, T_s)
        T_r = np.interp(self.z, self.orig_z, T_r)
        #return deltaT_interp, T_k_interp,

        return deltaT, T_K, T_s, T_r
