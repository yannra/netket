import netket as nk
import numpy as np

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
        sweep_by_bonds = False,
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
                    self.optimizer.sr._x0 = None
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
