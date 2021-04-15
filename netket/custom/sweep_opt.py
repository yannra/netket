import netket as nk
import numpy as np
from .linear_method import LinMethod
from .stabilised_sr import SRStab
from .stabilised_lin_method import LinMethodStab

class SweepOpt(nk.Vmc):
    def __init__(
        self,
        hamiltonian,
        sampler,
        optimizer,
        n_samples,
        n_discard=None,
        sr=None,
        max_opt = 3000,
        sweep_by_bonds = True,
    ):
        super().__init__(hamiltonian, sampler, optimizer, n_samples, n_discard=n_discard, sr=sr)
        self.max_opt = max_opt

        self.opt_arr = np.zeros(self._sampler._machine._epsilon.size, dtype=bool)
        self.opt_arr[:self.max_opt] = True
        self.max_id = min(self.max_opt, self.opt_arr.size)
        self.sweep_by_bonds = sweep_by_bonds

    def iter(self, n_steps, step=1):
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                if self.sweep_by_bonds:
                    opt_tensor = np.zeros(self._sampler._machine._epsilon.shape, dtype=bool)
                    arr_count = 0
                    for k in range(self._sampler._machine._epsilon.shape[1]):
                        for j in range(self._sampler._machine._epsilon.shape[0]):
                            for l in range(self._sampler._machine._epsilon.shape[2]):
                                opt_tensor[j,k,l] = self.opt_arr[arr_count]
                                arr_count += 1
                else:
                    opt_tensor = self.opt_arr.reshape(self._sampler._machine._epsilon.shape)

                self._sampler._machine.change_opt_ids(opt_tensor)
                if self.sr is not None and self.max_opt < self.opt_arr.size:
                    self.sr._x0 = None
                dp = self._forward_and_backward()
                self.update_parameters(dp)
                self.opt_arr.fill(False)
                self.opt_arr[self.max_id:(self.max_id+self.max_opt)] = True
                if self.max_id + self.max_opt > self.opt_arr.size:
                    self.opt_arr[:(self.max_id + self.max_opt - self.opt_arr.size)] = True
                    self.max_id = min((self.max_id + self.max_opt - self.opt_arr.size), self.opt_arr.size)
                else:
                    self.max_id = min((self.max_id + self.max_opt), self.opt_arr.size)
                if i == 0:
                    yield self.step_count


class SweepOptLinMethod(LinMethod):
    def __init__(
        self,
        hamiltonian,
        sampler,
        optimizer,
        n_samples,
        shift = 1,
        epsilon = 0.5,
        update_shift = True,
        corr_samp = None,
        rescale_update = True,
        n_discard=None,
        max_opt = 3000,
        sweep_by_bonds = True,
    ):
        super().__init__(hamiltonian, sampler, optimizer, n_samples, shift=shift, epsilon=epsilon,
                         update_shift=update_shift, corr_samp=corr_samp, rescale_update=rescale_update,
                         n_discard=n_discard)
        self.max_opt = max_opt

        self.opt_arr = np.zeros(self._sampler._machine._epsilon.size, dtype=bool)
        self.opt_arr[:self.max_opt] = True
        self.max_id = min(self.max_opt, self.opt_arr.size)
        self.sweep_by_bonds = sweep_by_bonds

    def iter(self, n_steps, step=1):
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                if self.sweep_by_bonds:
                    opt_tensor = np.zeros(self._sampler._machine._epsilon.shape, dtype=bool)
                    arr_count = 0
                    for k in range(self._sampler._machine._epsilon.shape[1]):
                        for j in range(self._sampler._machine._epsilon.shape[0]):
                            for l in range(self._sampler._machine._epsilon.shape[2]):
                                opt_tensor[j,k,l] = self.opt_arr[arr_count]
                                arr_count += 1
                else:
                    opt_tensor = self.opt_arr.reshape(self._sampler._machine._epsilon.shape)

                self._sampler._machine.change_opt_ids(opt_tensor)
                dp = self._forward_and_backward()
                self.update_parameters(dp)
                self.opt_arr.fill(False)
                self.opt_arr[self.max_id:(self.max_id+self.max_opt)] = True
                if self.max_id + self.max_opt > self.opt_arr.size:
                    self.opt_arr[:(self.max_id + self.max_opt - self.opt_arr.size)] = True
                    self.max_id = min((self.max_id + self.max_opt - self.opt_arr.size), self.opt_arr.size)
                else:
                    self.max_id = min((self.max_id + self.max_opt), self.opt_arr.size)
                if i == 0:
                    yield self.step_count


class SweepOptStabSR(SRStab):
    def __init__(
        self,
        hamiltonian,
        sampler,
        n_samples,
        sr,
        diag_shift = 0.01,
        time_step = 0.02,
        corr_samp = None,
        n_discard=None,
        search_radius=3,
        par_samples=4,
        max_opt = 3000,
        sweep_by_bonds = True,
    ):
        super().__init__(hamiltonian, sampler, n_samples, sr, diag_shift=diag_shift, time_step=time_step,
                         corr_samp=corr_samp, n_discard=n_discard, search_radius=search_radius, par_samples=par_samples)
        self.max_opt = max_opt

        self.opt_arr = np.zeros(self._sampler._machine._epsilon.size, dtype=bool)
        self.opt_arr[:self.max_opt] = True
        self.max_id = min(self.max_opt, self.opt_arr.size)
        self.sweep_by_bonds = sweep_by_bonds

    def iter(self, n_steps, step=1):
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                if self.sweep_by_bonds:
                    opt_tensor = np.zeros(self._sampler._machine._epsilon.shape, dtype=bool)
                    arr_count = 0
                    for k in range(self._sampler._machine._epsilon.shape[1]):
                        for j in range(self._sampler._machine._epsilon.shape[0]):
                            for l in range(self._sampler._machine._epsilon.shape[2]):
                                opt_tensor[j,k,l] = self.opt_arr[arr_count]
                                arr_count += 1
                else:
                    opt_tensor = self.opt_arr.reshape(self._sampler._machine._epsilon.shape)

                self._sampler._machine.change_opt_ids(opt_tensor)
                if self.sr is not None and self.max_opt < self.opt_arr.size:
                    self.sr._x0 = None
                dp = self._forward_and_backward()
                self.update_parameters(dp)
                self.opt_arr.fill(False)
                self.opt_arr[self.max_id:(self.max_id+self.max_opt)] = True
                if self.max_id + self.max_opt > self.opt_arr.size:
                    self.opt_arr[:(self.max_id + self.max_opt - self.opt_arr.size)] = True
                    self.max_id = min((self.max_id + self.max_opt - self.opt_arr.size), self.opt_arr.size)
                else:
                    self.max_id = min((self.max_id + self.max_opt), self.opt_arr.size)
                if i == 0:
                    yield self.step_count

class SweepOptStabLinMethod(LinMethodStab):
    def __init__(
        self,
        hamiltonian,
        sampler,
        n_samples,
        sr,
        diag_shift = 0.01,
        time_step = 0.02,
        corr_samp = None,
        n_discard=None,
        search_radius=3,
        par_samples=4,
        max_opt = 3000,
        sweep_by_bonds = True,
    ):
        super().__init__(hamiltonian, sampler, n_samples, sr, diag_shift=diag_shift, time_step=time_step,
                         corr_samp=corr_samp, n_discard=n_discard, search_radius=search_radius, par_samples=par_samples)
        self.max_opt = max_opt

        self.opt_arr = np.zeros(self._sampler._machine._epsilon.size, dtype=bool)
        self.opt_arr[:self.max_opt] = True
        self.max_id = min(self.max_opt, self.opt_arr.size)
        self.sweep_by_bonds = sweep_by_bonds

    def iter(self, n_steps, step=1):
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                if self.sweep_by_bonds:
                    opt_tensor = np.zeros(self._sampler._machine._epsilon.shape, dtype=bool)
                    arr_count = 0
                    for k in range(self._sampler._machine._epsilon.shape[1]):
                        for j in range(self._sampler._machine._epsilon.shape[0]):
                            for l in range(self._sampler._machine._epsilon.shape[2]):
                                opt_tensor[j,k,l] = self.opt_arr[arr_count]
                                arr_count += 1
                else:
                    opt_tensor = self.opt_arr.reshape(self._sampler._machine._epsilon.shape)

                self._sampler._machine.change_opt_ids(opt_tensor)
                if self.sr is not None and self.max_opt < self.opt_arr.size:
                    self.sr._x0 = None
                dp = self._forward_and_backward()
                self.update_parameters(dp)
                self.opt_arr.fill(False)
                self.opt_arr[self.max_id:(self.max_id+self.max_opt)] = True
                if self.max_id + self.max_opt > self.opt_arr.size:
                    self.opt_arr[:(self.max_id + self.max_opt - self.opt_arr.size)] = True
                    self.max_id = min((self.max_id + self.max_opt - self.opt_arr.size), self.opt_arr.size)
                else:
                    self.max_id = min((self.max_id + self.max_opt), self.opt_arr.size)
                if i == 0:
                    yield self.step_count