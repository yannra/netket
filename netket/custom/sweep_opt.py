import netket as nk
import numpy as np
from .linear_method import LinMethod
from .stabilised_sr import SRStab
from .stabilised_lin_method import LinMethodStab

from netket.stats import (
    statistics as _statistics,
    mean as _mean,
    sum_inplace as _sum_inplace,
)

from netket.utils import (
    MPI_comm as _MPI_comm,
    n_nodes as _n_nodes,
    node_number as _rank
)

from netket.vmc_common import info, tree_map


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
        check_improvement = True
    ):
        super().__init__(hamiltonian, sampler, optimizer, n_samples, n_discard=n_discard, sr=sr)
        self.max_opt = max_opt

        self.opt_arr = np.zeros(self._sampler._machine._epsilon.size, dtype=bool)
        self.opt_arr[:self.max_opt] = True
        self.max_id = min(self.max_opt, self.opt_arr.size)
        self.sweep_by_bonds = sweep_by_bonds
        self._valid_par = None
        assert(isinstance(self.optimizer, nk.optimizer.numpy.Sgd))
        self._default_timestep = self.optimizer._learning_rate
        if self._sr is not None:
            self._default_shift = self._sr._diag_shift
        self._check_improvement = check_improvement
        self._previous_mean = None
        self._previous_error = None

    def iter(self, n_steps, step=1):
        for count in range(0, n_steps, step):
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
                try:
                    dp = self._forward_and_backward()
                    if self._valid_par is None:
                        self._valid_par = self._sampler._machine._epsilon.copy()
                    else:
                        np.copyto(self._valid_par, self._sampler._machine._epsilon)
                    self._previous_mean = self._loss_stats.mean.real
                    self._previous_error = self._loss_stats.error_of_mean
                    if self._sr is not None:
                        self._sr._diag_shift = self._default_shift
                    self.optimizer._learning_rate = self._default_timestep

                    self.opt_arr.fill(False)
                    self.opt_arr[self.max_id:(self.max_id+self.max_opt)] = True
                    if self.max_id + self.max_opt > self.opt_arr.size:
                        self.opt_arr[:(self.max_id + self.max_opt - self.opt_arr.size)] = True
                        self.max_id = min((self.max_id + self.max_opt - self.opt_arr.size), self.opt_arr.size)
                    else:
                        self.max_id = min((self.max_id + self.max_opt), self.opt_arr.size)
                except:
                    print(_rank, "reset applied", flush = True)
                    assert(self._valid_par is not None)
                    print(_rank, abs(self._sampler._machine._epsilon - self._valid_par).max(), abs(self._sampler._machine._epsilon).max(), abs(self._valid_par).max(), flush = True)
                    np.copyto(self._sampler._machine._epsilon, self._valid_par)
                    np.copyto(self._sampler._machine._opt_params, self._sampler._machine._epsilon[self._sampler._machine._der_ids >= 0])
                    self.optimizer._learning_rate /= 2
                    if self._sr is not None:
                        self._sr._diag_shift *= 2
                    if _rank == 0:
                        print(count, "reset applied, learning rate:", self.optimizer._learning_rate, flush = True)
                        if self._sr is not None:
                            print("diag shift:", self._sr._diag_shift, flush=True)
                    check_improvement = self._check_improvement
                    self._check_improvement = False
                    dp = self._forward_and_backward()
                    self._previous_mean = self._loss_stats.mean.real
                    self._previous_error = self._loss_stats.error_of_mean
                    self._check_improvement = check_improvement

                if i == 0:
                    yield self.step_count
                self.update_parameters(dp)


    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self._sampler.reset()

        # Burnout phase
        self._sampler.generate_samples(self._n_discard)

        # Generate samples and store them
        self._samples = self._sampler.generate_samples(
            self._n_samples_node, samples=self._samples
        )

        # Compute the local energy estimator and average Energy
        eloc, self._loss_stats = self._get_mc_stats(self._ham)

        assert(self._loss_stats.mean.imag < 1.0)

        if self._check_improvement and self._previous_mean is not None:
            assert((self._loss_stats.mean.real - self._loss_stats.error_of_mean) < (self._previous_mean + self._previous_error))



        # Center the local energy
        eloc -= _mean(eloc)

        samples_r = self._samples.reshape((-1, self._samples.shape[-1]))
        eloc_r = eloc.reshape(-1, 1)

        # Perform update
        if self._sr:
            if self._sr.onthefly:

                self._grads = self._machine.vector_jacobian_prod(
                    samples_r, eloc_r / self._n_samples, self._grads
                )

                self._grads = tree_map(_sum_inplace, self._grads)

                self._dp = self._sr.compute_update_onthefly(
                    samples_r, self._grads, self._dp
                )

            else:
                # When using the SR (Natural gradient) we need to have the full jacobian
                self._grads, self._jac = self._machine.vector_jacobian_prod(
                    samples_r,
                    eloc_r / self._n_samples,
                    self._grads,
                    return_jacobian=True,
                )

                self._grads = tree_map(_sum_inplace, self._grads)

                self._dp = self._sr.compute_update(self._jac, self._grads, self._dp)

        else:
            # Computing updates using the simple gradient
            self._grads = self._machine.vector_jacobian_prod(
                samples_r, eloc_r / self._n_samples, self._grads
            )

            self._grads = tree_map(_sum_inplace, self._grads)

            #  if Real pars but complex gradient, take only real part
            # not necessary for SR because sr already does it.
            if not self._machine.has_complex_parameters:
                self._dp = tree_map(lambda x: x.real, self._grads)
            else:
                self._dp = self._grads

        return self._dp


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
                if i == 0:
                    yield self.step_count
                self.update_parameters(dp)
                self.opt_arr.fill(False)
                self.opt_arr[self.max_id:(self.max_id+self.max_opt)] = True
                if self.max_id + self.max_opt > self.opt_arr.size:
                    self.opt_arr[:(self.max_id + self.max_opt - self.opt_arr.size)] = True
                    self.max_id = min((self.max_id + self.max_opt - self.opt_arr.size), self.opt_arr.size)
                else:
                    self.max_id = min((self.max_id + self.max_opt), self.opt_arr.size)


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
                if i == 0:
                    yield self.step_count
                self.update_parameters(dp)
                self.opt_arr.fill(False)
                self.opt_arr[self.max_id:(self.max_id+self.max_opt)] = True
                if self.max_id + self.max_opt > self.opt_arr.size:
                    self.opt_arr[:(self.max_id + self.max_opt - self.opt_arr.size)] = True
                    self.max_id = min((self.max_id + self.max_opt - self.opt_arr.size), self.opt_arr.size)
                else:
                    self.max_id = min((self.max_id + self.max_opt), self.opt_arr.size)

class SweepOptStabLinMethod(LinMethodStab):
    def __init__(
        self,
        hamiltonian,
        sampler,
        n_samples,
        diag_shift = 0.01,
        time_step = 1.0,
        corr_samp = None,
        n_discard=None,
        search_radius=3,
        par_samples=10,
        max_opt = 3000,
        sweep_by_bonds = True,
    ):
        super().__init__(hamiltonian, sampler, n_samples, diag_shift=diag_shift, time_step=time_step,
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
                if i == 0:
                    yield self.step_count
                self.update_parameters(dp)
                self.opt_arr.fill(False)
                self.opt_arr[self.max_id:(self.max_id+self.max_opt)] = True
                if self.max_id + self.max_opt > self.opt_arr.size:
                    self.opt_arr[:(self.max_id + self.max_opt - self.opt_arr.size)] = True
                    self.max_id = min((self.max_id + self.max_opt - self.opt_arr.size), self.opt_arr.size)
                else:
                    self.max_id = min((self.max_id + self.max_opt), self.opt_arr.size)