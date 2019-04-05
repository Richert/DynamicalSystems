import numpy as np
import matplotlib.pyplot as plt


class QIFMicro:

    def __init__(self, N=1e3, J=15.0, eta_mean=-5.0, eta_fwhm=1.0, tau=0.02, alpha=0.0, tau_a=0.1,
                 v_th=1e2, v_0=-2.0):

        self.N = int(N)
        self.J = J * np.sqrt(eta_fwhm)
        self.eta = eta_mean + eta_fwhm*np.tan((np.pi*0.5)*(2*np.arange(1, N+1)-N-1)/(N+1))
        self.eta_fwhm = eta_fwhm
        self.eta_mean = eta_mean * eta_fwhm
        self.tau = tau
        self.alpha = alpha
        self.tau_a = tau_a
        self.v_th = v_th
        self.v = np.zeros((self.N,)) + v_0
        self.e = np.zeros_like(self.v)
        self.a = np.zeros_like(self.v)
        self.r = 0.0

    def run(self, T, dt, I=None, dts=None, **kwargs):

        if not dts:
            dts = dt
        store = int(dts/dt)
        steps_store = int(T/dts)
        steps_sim = int(T/dt)

        v_col = np.zeros((steps_store+1,))
        r_col = np.zeros_like(v_col)
        spikes = np.zeros((steps_sim, self.N))
        if I is None:
            I = np.zeros((steps_sim,))

        wait = np.zeros_like(self.v)
        wait_min = np.zeros_like(wait)
        spike_time = np.zeros_like(self.v)

        v_tmp = []
        r_tmp = []
        s = 0
        for step in range(steps_sim):

            # calculate firing rate
            wait = np.max([wait-1, wait_min], axis=0)
            mask = 1.0 * (wait == 0.)
            spike = 1.0 * (spike_time == 1.0)
            self.r = np.mean(spike) / dt

            # update state variables
            self.update(dt, mask, I[step])

            # calculate new spikes
            spike_start = self.v > self.v_th
            #if any(spike_start):
            wait[spike_start] = np.rint(2.*self.tau/(self.v[spike_start]*dt))
            spike_time[spike_start] = np.rint(1.*self.tau / (self.v[spike_start] * dt))
            spike_time = np.max([spike_time - 1, wait_min], axis=0)
            self.v[spike_start] = -self.v[spike_start]

            # store variables
            spikes[step, :] = spike
            v_tmp.append(np.mean(self.v))
            r_tmp.append(self.r)
            if np.mod(step, store) == 0:
                v_col[s] = np.mean(v_tmp, axis=0)
                r_col[s] = np.mean(r_tmp, axis=0)
                s += 1
                v_tmp.clear()
                r_tmp.clear()

        return v_col, r_col, spikes

    def update(self, dt, mask, I=0.0):

        v_delta = (self.v**2 + self.eta + I) / self.tau + self.J*(1.0 - np.arctan(self.e))*self.r
        e_delta = self.a
        a_delta = self.alpha*self.r/self.tau_a - 2*self.a/self.tau_a - self.e/self.tau_a**2

        self.v += dt * mask * v_delta
        self.e += dt * e_delta
        self.a += dt * a_delta


class QIFMacro:

    def __init__(self, J=15.0, eta_mean=-5.0, eta_fwhm=1.0, tau=0.02, alpha=0.0, tau_a=0.1, v_0=-2.0, r_0=0.0, e_0=0.0,
                 a_0=0.0):

        self.J = J * np.sqrt(eta_fwhm)
        self.eta = eta_mean
        self.Delta = eta_fwhm
        self.tau = tau
        self.alpha = alpha
        self.tau_a = tau_a
        self.v = v_0
        self.e = e_0
        self.a = a_0
        self.r = r_0

    def run(self, T, dt, I=None, dts=None, **kwargs):

        if not dts:
            dts = dt
        store = int(dts/dt)
        steps_store = int(T/dts)
        steps_sim = int(T/dt)

        v_col = np.zeros((steps_store+1,))
        r_col = np.zeros_like(v_col)
        e_col = np.zeros_like(v_col)
        if I is None:
            I = np.zeros((steps_sim,))

        v_tmp = []
        r_tmp = []
        e_tmp = []
        s = 0
        for step in range(steps_sim):

            # update state variables
            self.update(dt, I[step])

            # update parameters
            self.update_params(kwargs, step)

            # store variables
            v_tmp.append(self.v)
            r_tmp.append(self.r)
            e_tmp.append(self.e)
            if np.mod(step, store) == 0:
                v_col[s] = np.mean(v_tmp, axis=0)
                r_col[s] = np.mean(r_tmp, axis=0)
                e_col[s] = np.mean(e_tmp, axis=0)
                s += 1
                v_tmp = []
                r_tmp = []
                e_tmp = []

        return r_col, v_col, e_col

    def update(self, dt, I=0.0):

        r_delta = self.Delta/(np.pi*self.tau**2) + 2.0*self.v*self.r/self.tau
        v_delta = (self.v**2 + self.eta + I) / self.tau \
                  + self.J*(1.0 - np.tanh(self.e))*self.r \
                  - self.tau*(np.pi*self.r)**2
        e_delta = self.a
        a_delta = self.alpha*self.r/self.tau_a - 2*self.a/self.tau_a - self.e/self.tau_a**2

        self.r += dt * r_delta
        self.v += dt * v_delta
        self.e += dt * e_delta
        self.a += dt * a_delta

    def update_params(self, param_dict, idx):

        for key, item in param_dict.items():
            setattr(self, key, item[idx])
