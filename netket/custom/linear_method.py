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

from netket.vmc_common import info, tree_map
from netket import Vmc
from netket.abstract_variational_driver import AbstractVariationalDriver
from threadpoolctl import threadpool_limits


class LinMethod(Vmc):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

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
        rescale_update=True,
        n_discard=None,
    ):
        super().__init__(hamiltonian, sampler, optimizer, n_samples, n_discard=n_discard)
        self._stab_shift = shift
        self._epsilon = epsilon
        self.update_shift = update_shift
        if corr_samp is None:
            corr_samples = int(n_samples * 0.35)
        else:
            corr_samples = corr_samp
        n_corr_samples_chain = int(math.ceil((corr_samples / self._batch_size)))
        self._n_corr_samples_node = int(math.ceil(n_corr_samples_chain / self.n_nodes))
        self._n_corr_samples = int(self._n_corr_samples_node * self._batch_size * self.n_nodes)
        self.linear_params = np.zeros(len(self.machine.parameters), dtype=bool)
        self.rescale_update = rescale_update

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

    def get_parameter_update(self, H, S, oks_mean, shift):
        par_change = np.zeros(H.shape[0]-1, dtype=complex)
        if _rank == 0:
            H_mat = H + np.eye(H.shape[0]) * shift
            H_mat[0,0] -= shift
            eigs = eig(H_mat, S)
            phys_sensible_ids = np.logical_and(abs(eigs[0].imag) < 1, abs(eigs[0]) < 1.e5)
            i = eigs[0][phys_sensible_ids].real.argsort()[0]
            dp = eigs[1][:,phys_sensible_ids][:,i]
            dp /= dp[0]
            N = np.zeros(len(self.linear_params), dtype=dp.dtype)
            dp_reduced = dp[1:]
            if self.rescale_update:
                S_reduced = S[1:,1:]
                non_lin = ~self.linear_params

                N[non_lin] = -(((1-self._epsilon) * S_reduced[non_lin, non_lin].dot(dp_reduced[non_lin]))/((1-self._epsilon)+ self._epsilon * np.sqrt(1+ dp_reduced[non_lin].conj().dot(S_reduced[non_lin, non_lin].dot(dp_reduced[non_lin]))))).conj()
                N[self.linear_params] = oks_mean[self.linear_params]
                par_change = (dp_reduced / (1 - N.dot(dp_reduced)))
            else:
                par_change = dp_reduced

        
        if _n_nodes > 1:
            _MPI_comm.Bcast(par_change, root=0)
            _MPI_comm.barrier()
        return par_change

    def correlated_en_estimation(self, samples, ref_amplitudes, amplitudes):
        ratios = np.exp(2 * (amplitudes-ref_amplitudes).real)
        return (_mean(_local_values(self._ham, self.machine, samples) * ratios)/_mean(ratios))

    def recalculate_shift(self, H_full, S_full, oks_mean):
        samples = None
        best_e = None

        energies = []
        shifts = []
        test_shift = self._stab_shift
        count = 0
        valid_result = False

        while not valid_result:
            try:
                dp = self.get_parameter_update(H_full, S_full, oks_mean, test_shift)
                self.machine.parameters += dp
                try:
                    self._sampler.generate_samples(self._n_discard)
                    samples = self._sampler.generate_samples(self._n_corr_samples_node)
                    samples = samples.reshape((-1, samples.shape[-1]))
                    ref_amplitudes = self.machine.log_val(samples)
                    amplitudes = ref_amplitudes.copy()
                    e_new = self.correlated_en_estimation(samples, ref_amplitudes, amplitudes).real
                    valid_result = True
                except:
                    valid_result = False
                self.machine.parameters -= dp
            except:
                valid_result = False

            if not valid_result:
                _MPI_comm.bcast(valid_result, root=_rank)
            
            _MPI_comm.barrier()

            if valid_result:
                energies.append(e_new)
                shifts.append(test_shift)
                best_e = e_new
                best_shift = test_shift
                best_dp = dp
            else:
                test_shift *= 10
            count += 1
            assert(count < 10)
        
        for shift in (test_shift/10, test_shift * 10):
            dp = self.get_parameter_update(H_full, S_full, oks_mean, shift)
            self.machine.parameters += dp

            try:
                amplitudes = self.machine.log_val(samples)
                e_new = self.correlated_en_estimation(samples, ref_amplitudes, amplitudes).real
                valid_result = True
            except:
                valid_result = False

            if not valid_result:
                _MPI_comm.bcast(valid_result, root=_rank)
            
            _MPI_comm.barrier()

            self.machine.parameters -= dp

            if valid_result:
                energies.append(e_new)
                shifts.append(shift)
                if e_new < best_e:
                    best_e = e_new
                    best_shift = shift
                    best_dp = dp

        
        # parabolic interpolation if central shift gives largest improvement
        if len(energies) == 3:
            if energies[0] < energies[1] and energies[0] < energies[2]:
                interpolation_num = (energies[1] - energies[0]) * (np.log(shifts[2]) - np.log(shifts[0]))**2
                interpolation_num -= (energies[2] - energies[0]) * (np.log(shifts[0]) - np.log(shifts[1]))**2

                interpolation_denom = (energies[1] - energies[0]) * (np.log(shifts[2]) - np.log(shifts[0]))
                interpolation_denom += (energies[2] - energies[0]) * (np.log(shifts[0]) - np.log(shifts[1]))

                interpolated_shift = np.exp(np.log(shifts[1]) + 0.5 * interpolation_num/interpolation_denom)

                dp = self.get_parameter_update(H_full, S_full, oks_mean, interpolated_shift)

                self.machine.parameters += dp

                valid_result = False

                try:
                    amplitudes = self.machine.log_val(samples)
                    e_new = self.correlated_en_estimation(samples, ref_amplitudes, amplitudes).real
                    valid_result = True
                except:
                    valid_result = False

                if not valid_result:
                    _MPI_comm.bcast(valid_result, root=_rank)
            
                _MPI_comm.barrier()

                self.machine.parameters -= dp

                if valid_result:
                    if e_new < best_e:
                        best_e = e_new
                        best_shift = interpolated_shift
                        best_dp = dp

        self._stab_shift = best_shift
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

        if self.update_shift:
            self._dp = self.recalculate_shift(H_full, S_full, oks_mean)
        else:
            self._dp = self.get_parameter_update(H_full, S_full, oks_mean, self._stab_shift)

        return -self._dp
