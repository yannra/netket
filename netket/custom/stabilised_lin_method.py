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


class LinMethodStab(Vmc):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self,
        hamiltonian,
        sampler,
        n_samples,
        diag_shift = 0.01,
        time_step = 1,
        corr_samp = None,
        n_discard=None,
        search_radius=5,
        par_samples=10
    ):
        optimizer = op.Sgd(sampler.machine, learning_rate=1.0)
        super().__init__(hamiltonian, sampler, optimizer, n_samples, n_discard=n_discard)
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
    
    def get_lin_method_matrices(self, samples):
        oks = self.machine.der_log(samples)
        loc_en, der_loc_en = local_values_with_der(self._ham, self.machine, samples)

        en_stats = _statistics(loc_en.T)

        with threadpool_limits(limits=1):
            E0 = en_stats.mean
            oks_mean = _mean(oks, axis=0)
            der_loc_en_mean = _mean(der_loc_en, axis=0)
            ok_loc_en_mean = _mean((oks.conj().T * loc_en).T, axis=0)

            n_samp = _sum_inplace(np.atleast_1d(oks.shape[0]))

            overlapmat = np.matmul(oks.conj().T, oks)
            overlapmat = _sum_inplace(overlapmat)
            overlapmat /= float(n_samp)
            S = overlapmat - np.outer(oks_mean.conj(), oks_mean)

            Gr = der_loc_en_mean - E0 * oks_mean

            Gc = ok_loc_en_mean - E0 * oks_mean.conj()

            h_matr = np.matmul(oks.conj().T, der_loc_en)/n_samp
            h_matr = _sum_inplace(h_matr)
            h_matr /= float(n_samp)
            h_matr -= np.outer(ok_loc_en_mean, oks_mean)
            h_matr -= np.outer(oks_mean.conj(), der_loc_en_mean)
            h_matr += np.outer(oks_mean.conj(), oks_mean)*E0

            S_full = np.block([[np.ones((1,1)), np.zeros((1, S.shape[0]))],[np.zeros((S.shape[0],1)), S]])
            H_full = np.block([[E0 * np.ones((1,1)), Gr.reshape((1, len(Gr)))],[Gc.reshape((len(Gc), 1)), h_matr]])

        return (S_full, H_full, oks_mean, en_stats)

    def get_parameter_update(self, H, S, oks_mean, shift, timestep):
        par_change = np.zeros(H.shape[0]-1, dtype=complex)
        if _rank == 0:
            H_mat = H + np.eye(H.shape[0]) * shift
            H_mat[0,0] -= shift
            eigs = eig(H_mat, S)
            phys_sensible_ids = np.logical_and(abs(eigs[0].imag) < 1, abs(eigs[0]) < 1.e5)
            i = eigs[0][phys_sensible_ids].real.argsort()[0]
            dp = eigs[1][:,phys_sensible_ids][:,i]
            dp /= dp[0]
            par_change = (-timestep*dp[1:]).copy()
        
        if _n_nodes > 1:
            _MPI_comm.Bcast(par_change, root=0)
            _MPI_comm.barrier()
        return par_change

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

    def _linesearch(self, S, H, oks_mean):
        samples = None
        best_e = None

        test_shift = self._diag_shift
        test_step = self._time_step
        count = 0
        valid_result = False

        while not valid_result:
            try:
                dp = self.get_parameter_update(H, S, oks_mean, test_shift, test_step)
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
            assert(count < 100)
        
        test_params = np.zeros((self._par_samples,2))
        
        if _rank == 0:
            # randomly sample new parameters from [central/radius, radius * central] (log-uniformly distributed)
            test_shifts = np.exp(np.random.rand(self._par_samples) * (np.log(best_shift*self._search_radius) - np.log(best_shift/self._search_radius)) + np.log(best_shift/self._search_radius))
            test_steps = np.exp(np.random.rand(self._par_samples) * (np.log(best_step*self._search_radius) - np.log(best_step/self._search_radius)) + np.log(best_step/self._search_radius))
            test_params[:,:] = np.array([[test_shifts[i], test_steps[i]] for i in range(self._par_samples)])

        _MPI_comm.Bcast(test_params, root=0)
        _MPI_comm.barrier()

        for parameters in test_params:
            try:
                dp = self.get_parameter_update(H, S, oks_mean, parameters[0], parameters[1])
                self.machine.parameters -= dp
                try:
                    amplitudes = self.machine.log_val(samples_r)
                    e_new = self.correlated_en_estimation(samples_r, ref_amplitudes, amplitudes).real
                    valid_result = True
                except:
                    valid_result = False

                self.machine.parameters += dp
            except:
                valid_result = False

            if valid_result:
                if e_new < best_e:
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

        samples_r = self._samples.reshape((-1, self._samples.shape[-1]))

        S_full, H_full, oks_mean, self._loss_stats = self.get_lin_method_matrices(samples_r)

        self._dp = self._linesearch(S_full, H_full, oks_mean)

        return self._dp
