import math

import netket as _nk

from netket.operator import local_values as _local_values
from netket.custom import local_values_with_der
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

import numpy as np
from scipy.linalg import eig

import netket.optimizer as op

from netket.vmc_common import info, tree_map
from netket import Vmc
from netket.abstract_variational_driver import AbstractVariationalDriver
from threadpoolctl import threadpool_limits


class SRStab(Vmc):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

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
        par_samples=4
    ):
        assert(not sr.onthefly)
        optimizer = op.Sgd(sampler.machine, learning_rate=1.0)
        super().__init__(hamiltonian, sampler, optimizer, n_samples, n_discard=n_discard, sr=sr)
        self._diag_shift = diag_shift
        self._time_step = time_step
        if corr_samp is None:
            corr_samples = int(n_samples * 0.35)
        else:
            corr_samples = corr_samp
        n_corr_samples_chain = int(math.ceil((corr_samples / self._batch_size)))
        self._n_corr_samples_node = int(math.ceil(n_corr_samples_chain / self.n_nodes))
        self._n_corr_samples = int(self._n_corr_samples_node * self._batch_size * self.n_nodes)
        self._search_radius = search_radius
        self._par_samples = par_samples

    def correlated_en_estimation(self, samples, ref_amplitudes, amplitudes):
        ratios = np.exp(2 * (amplitudes-ref_amplitudes).real)
        loc_vals = _local_values(self._ham, self.machine, samples)
        val = (_mean(loc_vals * ratios)/_mean(ratios))
        assert(np.isfinite(val))
        return val

    def en_estimation(self, samples_r, samples):
        loc = _local_values(self._ham, self.machine, samples_r).reshape(samples.shape[0:2])
        stat = _statistics(loc)
        assert(np.isfinite(stat.mean))
        return stat

    def _linesearch(self):
        samples = None
        best_e = None

        test_shift = self._diag_shift
        test_step = self._time_step
        count = 0
        valid_result = False

        while not valid_result:
            try:
                self._sr._diag_shift = test_shift
                dp = test_step * self._sr.compute_update(self._jac, self._grads, self._dp)
                self.machine.parameters -= dp
                try:
                    self._sampler.generate_samples(self._n_discard)
                    samples = self._sampler.generate_samples(self._n_corr_samples_node)
                    samples_r = samples.reshape((-1, samples.shape[-1]))
                    ref_amplitudes = self.machine.log_val(samples_r)
                    stats = self.en_estimation(samples_r, samples)
                    # TODO: work out good criterion
                    if ((stats.mean.real) < self._loss_stats.mean.real + 2*self._loss_stats.error_of_mean) and abs(stats.mean.imag) < 1:
                        valid_result = True
                    else:
                        valid_result = False
                except:
                    valid_result = False
                self.machine.parameters += dp
            except:
                valid_result = False

            if valid_result:
                best_e = stats.mean.real
                best_shift = test_shift
                best_step = test_step
                best_dp = dp
            else:
                test_shift *= 1.5
                test_step /= 1.5
            count += 1
            if count > 3:
                print(count, self._loss_stats.mean.real, test_shift, test_step, flush=True)
            assert(count < 100)
        
        test_params = np.zeros((self._par_samples,2))
        
        if _rank == 0:
            # randomly sample new parameters from [central/5, 5 * central] (log-uniformly distributed)
            test_shifts = np.exp(np.random.rand(self._par_samples) * (np.log(best_shift*self._search_radius) - np.log(best_shift/self._search_radius)) + np.log(best_shift/self._search_radius))
            test_steps = np.exp(np.random.rand(self._par_samples) * (np.log(best_step*self._search_radius) - np.log(best_step/self._search_radius)) + np.log(best_step/self._search_radius))
            test_params[:,:] = np.array([[test_shifts[i], test_steps[i]] for i in range(self._par_samples)])

        _MPI_comm.Bcast(test_params, root=0)
        _MPI_comm.barrier()

        for parameters in test_params:
            try:
                self._sr._diag_shift = parameters[0]
                dp = parameters[1] * self._sr.compute_update(self._jac, self._grads, self._dp)
                self.machine.parameters -= dp
                try:
                    amplitudes = self.machine.log_val(samples_r)
                    stats = self.correlated_en_estimation(samples_r, ref_amplitudes, amplitudes)
                    e_new = stats.real
                    valid_result = True
                except:
                    valid_result = False

                self.machine.parameters += dp
            except:
                valid_result = False

            if valid_result:
                if e_new < best_e and abs(stats.imag) < 1:
                    best_e = e_new
                    best_shift = parameters[0]
                    best_step = parameters[1]
                    best_dp = dp

        self._diag_shift = best_shift
        self._time_step = best_step
        return best_dp

    def _forward_and_backward(self):
        self._sampler.reset()

        # Burnout phase
        self._sampler.generate_samples(self._n_discard)

        # Generate samples and store them
        self._samples = self._sampler.generate_samples(
            self._n_samples_node, samples=self._samples
        )

        # Compute the local energy estimator and average Energy
        eloc, self._loss_stats = self._get_mc_stats(self._ham)

        # Center the local energy
        eloc -= _mean(eloc)

        samples_r = self._samples.reshape((-1, self._samples.shape[-1]))
        eloc_r = eloc.reshape(-1, 1)

        # When using the SR (Natural gradient) we need to have the full jacobian
        self._grads, self._jac = self._machine.vector_jacobian_prod(
            samples_r,
            eloc_r / self._n_samples,
            self._grads,
            return_jacobian=True,
        )

        self._grads = tree_map(_sum_inplace, self._grads)

        self._dp = self._linesearch()

        return self._dp
