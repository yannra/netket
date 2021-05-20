import math

import netket as _nk

import netket.optimizer as op

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
from scipy.linalg import lstsq

from netket.vmc_common import info, tree_map
from netket import Vmc
from netket.abstract_variational_driver import AbstractVariationalDriver
from threadpoolctl import threadpool_limits


class NewtonMethod(Vmc):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self,
        hamiltonian,
        sampler,
        n_samples,
        shift = 500,
        corr_samp = None,
        n_discard=None,
        reestimate_shift=True,
        SRH = False,
        symmetrise_hessian=False
    ):
        super().__init__(hamiltonian, sampler, op.Sgd(sampler.machine, learning_rate=1.0), n_samples, n_discard=n_discard)
        self._stab_shift = shift
        if corr_samp is None:
            corr_samples = int(n_samples * 0.35)
        else:
            corr_samples = corr_samp
        n_corr_samples_chain = int(math.ceil((corr_samples / self._batch_size)))
        self._n_corr_samples_node = int(math.ceil(n_corr_samples_chain / self.n_nodes))
        self._n_corr_samples = int(self._n_corr_samples_node * self._batch_size * self.n_nodes)
        self.reestimate_shift = reestimate_shift
        self.SRH = SRH
        self.symmetrise = symmetrise_hessian

    def get_grad_hessian(self, samples, symmetrise):
        with threadpool_limits(limits=1):
            n_samples = samples.shape[0]

            oks = self.machine.der_log(samples)

            loc_en, der_loc_en = local_values_with_der(self._ham, self.machine, samples)

            oks_derivative = self.machine.hess(samples).conj()
        
            en_stats = _statistics(loc_en.T)
            if self.machine.has_complex_parameters:
                oks = np.concatenate((oks, 1j*oks), axis=1)
                der_loc_en = np.concatenate((der_loc_en, 1j*der_loc_en), axis=1)
                oks_derivative = np.block([[oks_derivative, -1.j*oks_derivative],[-1.j*oks_derivative,-oks_derivative]])


            E0 = en_stats.mean
            oks_mean = _mean(oks, axis=0)

            overlapmat = np.matmul(oks.conj().T, oks)
            overlapmat = _sum_inplace(overlapmat)
            overlapmat /= float(n_samples)
            S = overlapmat.real - np.outer(oks_mean.real, oks_mean.real)

            n_samples = _sum_inplace(np.atleast_1d(oks.shape[0]))

            delta_E = loc_en - E0

            gradient = 2*(oks.conj().T.dot(delta_E/n_samples)).real

            A = 2 * _mean((oks_derivative.T * delta_E).real, axis=2)

            B = np.matmul((oks.conj().T * delta_E).real, oks.real)
            B = _sum_inplace(B)
            B /= float(n_samples)
            tmp = np.outer((_mean(oks.conj().T * delta_E, axis=1)).real, oks_mean.real)
            B -= tmp + tmp.T
            B *= 4

            tmp = np.matmul(oks.conj().T, der_loc_en)
            tmp = _sum_inplace(tmp)
            tmp /= float(n_samples)
            C = tmp.real
            tmp = np.matmul(loc_en * oks.conj().T, oks)
            tmp = _sum_inplace(tmp)
            tmp /= float(n_samples)
            C -= tmp.real
            C *= 2

            if symmetrise:
                he = 0.5 *(A+B+C + A.T + B.T + C.T)
            else:
                he = A + B + C

        return (he, gradient, S, en_stats)


    def get_parameter_update(self, hessian, grad, S, shift):
        n_par = self.machine.n_par
        par_change = np.zeros(n_par, dtype=complex)
        if _rank == 0:
            if self.SRH:
                hess = hessian + S * shift
            else:
                hess = hessian + np.eye(hessian.shape[0]) * shift
            dp, _, _, _ = lstsq(hess, grad)

            if self.machine.has_complex_parameters:
                par_change = dp[:n_par] + 1.j * dp[n_par:]
            else:
                par_change = dp

        if _n_nodes > 1:
            _MPI_comm.Bcast(par_change, root=0)
            _MPI_comm.barrier()
        return par_change

    def correlated_en_estimation(self, samples, ref_amplitudes, amplitudes):
        ratios = np.exp(2 * (amplitudes-ref_amplitudes).real)
        val = (_mean(_local_values(self._ham, self.machine, samples) * ratios)/_mean(ratios))
        assert(np.isfinite(val))
        return val

    def en_estimation(self, samples_r, samples):
        loc = _local_values(self._ham, self.machine, samples_r).reshape(samples.shape[0:2])
        stat = _statistics(loc)
        assert(np.isfinite(stat.mean))
        return stat

    def _linesearch(self, hessian, grad, S):
        samples = None
        best_e = None

        energies = []
        shifts = []
        test_shift = self._stab_shift
        count = 0
        valid_result = False
        init_shift = self._stab_shift

        while not valid_result:
            try:
                dp = self.get_parameter_update(hessian, grad, S, test_shift)
                self.machine.parameters -= dp
                try:
                    self._sampler.generate_samples(self._n_discard)
                    samples = self._sampler.generate_samples(self._n_corr_samples_node)
                    samples_r = samples.reshape((-1, samples.shape[-1]))
                    ref_amplitudes = self.machine.log_val(samples_r)
                    stats = self.en_estimation(samples_r, samples)
                    # TODO: work out good criterion
                    if (((stats.mean.real - stats.error_of_mean) < (self._loss_stats.mean.real + self._loss_stats.error_of_mean)) or
                        np.allclose(stats.mean.real, self._loss_stats.mean.real)) and abs(stats.mean.imag) < 1 :
                        valid_result = True
                    else:
                        print(stats, self._loss_stats)
                        valid_result = False
                except:
                    valid_result = False
                self.machine.parameters += dp
            except:
                valid_result = False

            if valid_result:
                energies.append(stats.mean.real)
                shifts.append(test_shift)
                best_e = stats.mean.real
                best_shift = test_shift
                best_dp = dp
            else:
                test_shift *= 2

            count += 1
            if count >= 100:
                # self._stab_shift = init_shift
                print("recalculate", flush=True)
                return 0.0
        
        if self.reestimate_shift:
            for shift in (test_shift/2, test_shift * 2):
                dp = self.get_parameter_update(hessian, grad, S, test_shift)
                self.machine.parameters -= dp

                try:
                    amplitudes = self.machine.log_val(samples_r)
                    e_new = self.correlated_en_estimation(samples_r, ref_amplitudes, amplitudes).real
                    valid_result = True
                except:
                    valid_result = False

                self.machine.parameters += dp

                if valid_result:
                    energies.append(e_new)
                    shifts.append(shift)
                    if e_new < best_e:
                        best_e = e_new
                        best_shift = shift
                        best_dp = dp

            # parabolic interpolation if central shift not worst energy
            if len(energies) == 3:
                if energies[0] < energies[1] or energies[0] < energies[2]:
                    interpolation_num = (energies[1] - energies[0]) * ((shifts[2]) - (shifts[0]))**2
                    interpolation_num -= (energies[2] - energies[0]) * ((shifts[0]) - (shifts[1]))**2

                    interpolation_denom = (energies[1] - energies[0]) * ((shifts[2]) - (shifts[0]))
                    interpolation_denom += (energies[2] - energies[0]) * ((shifts[0]) - (shifts[1]))

                    interpolated_shift = ((shifts[1]) + 0.5 * interpolation_num/interpolation_denom)

                    print(test_shift, interpolated_shift, flush=True)

                    dp = self.get_parameter_update(hessian, grad, S, interpolated_shift)

                    self.machine.parameters -= dp

                    valid_result = False

                    try:
                        amplitudes = self.machine.log_val(samples_r)
                        e_new = self.correlated_en_estimation(samples_r, ref_amplitudes, amplitudes).real
                        valid_result = True
                    except:
                        valid_result = False

                    self.machine.parameters += dp

                    if valid_result:
                        print("success", e_new, best_e,  flush=True)
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

        hessian, grad, S, self._loss_stats = self.get_grad_hessian(samples_r, self.symmetrise)

        self._dp = self._linesearch(hessian, grad, S)

        return self._dp
