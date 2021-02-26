from .abstract_machine import AbstractMachine
from netket.graph import AbstractGraph
import numpy as _np
from numba import jit
import time
from netket.utils import (
    MPI_comm as _MPI_comm,
    n_nodes as _n_nodes,
    node_number as _rank
)


class QGPS(AbstractMachine):
    r"""
    The QGPS ansatz.
    """

    def __init__(
        self,
        hilbert,
        epsilon=None,
        n_bond=None,
        automorphisms=None,
        spin_flip_sym=False,
        cluster_ids=None,
        dtype=complex
    ):
        n = hilbert.size

        if dtype is not float and dtype is not complex:
            raise TypeError("dtype must be either float or complex")

        self._dtype = dtype
        self._npdtype = _np.complex128 if dtype is complex else _np.float64

        if automorphisms is None:
            self._Smap = _np.zeros((1, n), dtype=_np.intp)
            k = 0
            for i in range(n):
                self._Smap[0, i] = k
                k += 1

        else:
            if isinstance(automorphisms, AbstractGraph):
                autom = _np.asarray(automorphisms.automorphisms())
            else:
                try:
                    autom = _np.asarray(automorphisms)
                    assert n == autom.shape[1]
                except:
                    raise RuntimeError("Cannot find a valid automorphism array.")

            self._Smap = autom
        
        if cluster_ids is not None:
            self._Smap = self._Smap[:, cluster_ids]

        cluster_size = self._Smap.shape[1]

        self._sym_spin_flip_sign = _np.ones(len(self._Smap), dtype=_np.int8)
        if spin_flip_sym:
            self._Smap = _np.append(self._Smap, self._Smap, axis=0)
            self._sym_spin_flip_sign = _np.append(self._sym_spin_flip_sign,
                                                  -self._sym_spin_flip_sign, axis=0)
        if epsilon is None:
            assert(n_bond is not None)
            self._epsilon = _np.zeros((cluster_size, n_bond, 2), dtype=self._npdtype)
        else:
            self._epsilon = epsilon.astype(self._npdtype)

        self.value_time = 0
        self.der_time = 0
        self._der_ids = _np.array(range(self._epsilon.size)).reshape(self._epsilon.shape)
        self._opt_params = self._epsilon[self._der_ids >= 0].copy()
        self._npar = self._opt_params.size

        super().__init__(hilbert, dtype=dtype)

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._opt_params.size
    
    def init_random_parameters(self, seed=None, sigma=0.1, start_from_uniform=True):
        epsilon = _np.ones(self._epsilon.shape, dtype=self._npdtype)

        if _rank == 0:
            rgen = _np.random.default_rng(seed)
            epsilon += rgen.normal(scale=sigma, size=epsilon.shape)
            if self._dtype == complex:
                epsilon += 1j*rgen.normal(scale=sigma, size=epsilon.shape)
            if start_from_uniform:
                epsilon[0,:,:] = 0.0

        if _n_nodes > 1:
            _MPI_comm.Bcast(epsilon, root=0)
            _MPI_comm.barrier()

        self._epsilon = epsilon
        self._opt_params = self._epsilon[self._der_ids >= 0].copy()
    
    def change_opt_ids(self, opt_ids):
        # TODO improve this
        count = 0
        for i in range(opt_ids.shape[0]):
            for j in range(opt_ids.shape[1]):
                for k in range(opt_ids.shape[2]):
                    if opt_ids[i,j,k]:
                        self._der_ids[i,j,k] = count
                        count += 1
                    else:
                        self._der_ids[i,j,k] = -1
        self._opt_params = self._epsilon[self._der_ids >= 0].copy()
        self._npar = self._opt_params.size

    def init_random_parameters_alt(self, seed=None, sigma=0.1):
        epsilon = _np.zeros(self._epsilon.shape, dtype=self._npdtype)

        if _rank == 0:
            rgen = _np.random.default_rng(seed)
            epsilon += rgen.normal(scale=sigma, size=epsilon.shape)
            if self._dtype == complex:
                epsilon += 1j*rgen.normal(scale=sigma, size=epsilon.shape)
            for i in range(epsilon.shape[0]):
                for j in range(epsilon.shape[1]):
                    epsilon[i, j, :] /= _np.max(abs(epsilon[i, j, :]))

        if _n_nodes > 1:
            _MPI_comm.Bcast(epsilon, root=0)
            _MPI_comm.barrier()

        self._epsilon = epsilon
        self._opt_params = self._epsilon[self._der_ids >= 0].copy()

    def log_val(self, x, out=None):
        r"""Computes the logarithm of the wave function for a batch of visible
        configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. The
                 length of `out` should be `x.shape[0]`.

        Returns:
            A complex number when `x` is a vector and vector when `x` is a
            matrix.
        """
        start = time.time()
        val = self._log_val_kernel(x, out, self._epsilon, self._Smap, self._sym_spin_flip_sign)
        self.value_time += time.time() - start
        return val

    def der_log(self, x, out=None):
        r"""Computes the gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`.
                `out` should be a matrix of shape `(v.shape[0], self.n_par)`.

        Returns:
            `out`
        """
        start = time.time()
        der = self._der_log_kernel(x, out, self._epsilon, self._npar, self._Smap, self._sym_spin_flip_sign, self._der_ids)
        self.der_time += time.time() - start
        return der

    @property
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        from collections import OrderedDict

        od = OrderedDict()
        if self._dtype is complex:
            od["epsilon_opt"] = self._opt_params.view()

        else:
            self._opt_paramsc = self._opt_params.astype(_np.complex128)
            self._opt_params = self._opt_paramsc.real.view()
            od["epsilon_opt"] = self._opt_paramsc.view()

        return od

    # TODO: Clean this up
    @property
    def parameters(self):
        return _np.concatenate(tuple(p.reshape(-1) for p in self.state_dict.values()))

    @parameters.setter
    def parameters(self, p):
        if p.shape != (self.n_par,):
            raise ValueError(
                "p has wrong shape: {}; expected ({},)".format(p.shape, self.n_par)
            )

        i = 0
        for x in map(lambda x: x.reshape(-1), self.state_dict.values()):
            _np.copyto(x, p[i : i + x.size])
            i += x.size
        self._epsilon[self._der_ids >= 0] = self._opt_params
    

class QGPSSumSym(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, cluster_ids=None, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         cluster_ids=cluster_ids, dtype=dtype)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, Smap, symSign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                arg = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerprod *= epsilon[i, w, 0]
                        else:
                            innerprod *= epsilon[i, w, 1]
                    arg += innerprod
                out[b] += _np.exp(arg)
            out[b] = _np.log(out[b])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_par, Smap, symSign, der_ids):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            value = _np.complex128(0.0)
            for t in range(Smap.shape[0]):
                argument = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerargument = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerargument *= epsilon[i, w, 0]
                        else:
                            innerargument *= epsilon[i, w, 1]
                    argument += innerargument
                prefactor = _np.exp(argument)

                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                if _np.abs(epsilon[i, w, 0]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 0]] += prefactor * derivative/epsilon[i, w, 0]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    out[b, der_ids[i, w, 0]] += prefactor * der
                        else:
                            if der_ids[i, w, 1] >= 0:
                                if _np.abs(epsilon[i, w, 1]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 1]] += prefactor * derivative/epsilon[i, w, 1]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    out[b, der_ids[i, w, 1]] += prefactor * der
                value += prefactor
            out[b, :] /= value
        return out


class QGPSProdSym(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, cluster_ids=None, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         cluster_ids=cluster_ids, dtype=dtype)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, Smap, symSign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerprod *= epsilon[i, w, 0]
                        else:
                            innerprod *= epsilon[i, w, 1]
                    out[b] += innerprod
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_par, Smap, symSign, der_ids):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                if _np.abs(epsilon[i, w, 0]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 0]] += derivative/epsilon[i, w, 0]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    out[b, der_ids[i, w, 0]] += der
                        else:
                            if der_ids[i, w, 1] >= 0:
                                if _np.abs(epsilon[i, w, 1]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 1]] += derivative/epsilon[i, w, 1]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    out[b, der_ids[i, w, 1]] += der
        return out


class QGPSExp(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, cluster_ids=None, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         cluster_ids=cluster_ids, dtype=dtype)

    def init_random_parameters(self, seed=None, sigma=0.1):
        epsilon = _np.zeros(self._epsilon.shape, dtype=self._npdtype)

        if _rank == 0:
            rgen = _np.random.default_rng(seed)
            epsilon[0,:,:] += rgen.normal(scale=sigma, size=(epsilon.shape[1], epsilon.shape[2]))
            if self._dtype == complex:
                epsilon[0,:,:] += 1j*rgen.normal(scale=sigma, size=(epsilon.shape[1], epsilon.shape[2]))

        if _n_nodes > 1:
            _MPI_comm.Bcast(epsilon, root=0)
            _MPI_comm.barrier()

        self._epsilon = epsilon


class QGPSSumSymExp(QGPSExp):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, cluster_ids=None, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         cluster_ids=cluster_ids, dtype=dtype)
    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, Smap, symSign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                arg = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    exparg = _np.complex128(0.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            exparg += epsilon[i, w, 0]
                        else:
                            exparg += epsilon[i, w, 1]
                    arg += _np.exp(exparg)
                out[b] += _np.exp(arg)
            out[b] = _np.log(out[b])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_par, Smap, symSign, der_ids):
        batch_size = x.shape[0]

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            value = _np.complex128(0.0)
            for t in range(Smap.shape[0]):
                argument = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerargument = _np.complex128(0.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerargument += epsilon[i, w, 0]
                        else:
                            innerargument += epsilon[i, w, 1]
                    argument += _np.exp(innerargument)
                prefactor = _np.exp(argument)

                for w in range(epsilon.shape[1]):
                    arg = _np.complex128(0.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            arg += epsilon[i, w, 0]
                        else:
                            arg += epsilon[i, w, 1]
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                out[b, der_ids[i, w, 0]] += prefactor * _np.exp(arg)
                        else:
                            if der_ids[i, w, 1] >= 0:
                                out[b, der_ids[i, w, 1]] += prefactor * _np.exp(arg)
                value += prefactor
            out[b, :] /= value
        return out


class QGPSProdSymExp(QGPSExp):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, cluster_ids=None, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         cluster_ids=cluster_ids, dtype=dtype)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, Smap, symSign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    arg = _np.complex128(0.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            arg += epsilon[i, w, 0]
                        else:
                            arg += epsilon[i, w, 1]
                    out[b] += _np.exp(arg)
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_par, Smap, symSign, der_ids):
        batch_size = x.shape[0]

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(0.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            derivative += epsilon[i, w, 0]
                        else:
                            derivative += epsilon[i, w, 1]
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                out[b, der_ids[i, w, 0]] += _np.exp(derivative)
                        else:
                            if der_ids[i, w, 1] >= 0:
                                out[b, der_ids[i, w, 1]] += _np.exp(derivative)
        return out


class QGPSPhaseSplit(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond_amplitude=None, n_bond_phase=None,
                 automorphisms=None, spin_flip_sym=False, cluster_ids=None):
        self.n_bond_amplitude = n_bond_amplitude

        if n_bond_phase is not None and n_bond_amplitude is not None:
            n_bond_total = n_bond_phase + n_bond_amplitude
            if epsilon is not None:
                assert(n_bond_total == epsilon.shape[1])
        else:
            assert(epsilon is not None)
            n_bond_total = epsilon.shape[1]
            if n_bond_phase is not None:
                if n_bond_amplitude is not None:
                    assert(n_bond_phase + n_bond_amplitude == n_bond_total)
                else:
                    self.n_bond_amplitude = n_bond_total - n_bond_phase
            else:
                self.n_bond_amplitude = _np.ceil(n_bond_total/2)

        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond_total,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         cluster_ids=cluster_ids, dtype=float)

    def log_val(self, x, out=None):
        r"""Computes the logarithm of the wave function for a batch of visible
        configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. The
                 length of `out` should be `x.shape[0]`.

        Returns:
            A complex number when `x` is a vector and vector when `x` is a
            matrix.
        """
        start = time.time()
        val = self._log_val_kernel(x, out, self._epsilon, self.n_bond_amplitude, 
                                   self._Smap, self._sym_spin_flip_sign)
        self.value_time += time.time() - start
        return val

    def der_log(self, x, out=None):
        r"""Computes the gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`.
                `out` should be a matrix of shape `(v.shape[0], self.n_par)`.

        Returns:
            `out`
        """
        start = time.time()
        der = self._der_log_kernel(x, out, self._epsilon, self.n_bond_amplitude,
                                   self._npar, self._Smap, self._sym_spin_flip_sign, self._der_ids)
        self.der_time += time.time() - start
        return der
    
    def init_random_parameters(self, seed=None, sigma=0.1, random_sign=False):
        epsilon = _np.ones(self._epsilon.shape, dtype=self._npdtype)

        if _rank == 0:
            rgen = _np.random.default_rng(seed)
            amp_bond = self.n_bond_amplitude
            eps_shape = epsilon.shape
            epsilon[:,:,:] += rgen.normal(scale=sigma, size=eps_shape)
            epsilon[0,:amp_bond,:] = 0.0
            epsilon[0,amp_bond:,:] = sigma
            if random_sign:
                epsilon *= (2*rgen.integers(0,2,size=epsilon.shape)-1)

        if _n_nodes > 1:
            _MPI_comm.Bcast(epsilon, root=0)
            _MPI_comm.barrier()

        self._epsilon = epsilon


class QGPSPhaseSplitSumSym(QGPSPhaseSplit):
    def __init__(self, hilbert, epsilon=None, n_bond_amplitude=None, n_bond_phase=None,
                 automorphisms=None, spin_flip_sym=False, cluster_ids=None):
        super().__init__(hilbert, epsilon=epsilon, n_bond_amplitude=n_bond_amplitude,
                         n_bond_phase=n_bond_phase, automorphisms=automorphisms,
                         spin_flip_sym=spin_flip_sym, cluster_ids=cluster_ids)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, n_bond_amplitude, Smap, symSign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                arg = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerprod *= epsilon[i, w, 0]
                        else:
                            innerprod *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        innerprod *= 1.0j
                    arg += innerprod
                out[b] += _np.exp(arg)
            out[b] = _np.log(out[b])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_bond_amplitude, n_par, Smap, symSign, der_ids):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            value = _np.complex128(0.0)
            for t in range(Smap.shape[0]):
                argument = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerargument = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerargument *= epsilon[i, w, 0]
                        else:
                            innerargument *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        innerargument *= 1.0j
                    argument += innerargument
                prefactor = _np.exp(argument)

                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        derivative *= 1.0j
                

                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                if _np.abs(epsilon[i, w, 0]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 0]] += prefactor * derivative/epsilon[i, w, 0]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    if w >= n_bond_amplitude:
                                        der *= 1.0j
                                    out[b, der_ids[i, w, 0]] += prefactor * der
                        else:
                            if der_ids[i, w, 1] >= 0:
                                if _np.abs(epsilon[i, w, 1]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 1]] += prefactor * derivative/epsilon[i, w, 1]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    if w >= n_bond_amplitude:
                                        der *= 1.0j
                                    out[b, der_ids[i, w, 1]] += prefactor * der
                value += prefactor
            out[b, :] /= value
        return out


class QGPSPhaseSplitProdSym(QGPSPhaseSplit):
    def __init__(self, hilbert, epsilon=None, n_bond_amplitude=None, n_bond_phase=None,
                 automorphisms=None, spin_flip_sym=False, cluster_ids=None):
        super().__init__(hilbert, epsilon=epsilon, n_bond_amplitude=n_bond_amplitude,
                         n_bond_phase=n_bond_phase, automorphisms=automorphisms,
                         spin_flip_sym=spin_flip_sym, cluster_ids=cluster_ids)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, n_bond_amplitude, Smap, symSign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerprod *= epsilon[i, w, 0]
                        else:
                            innerprod *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        innerprod *= 1.0j
                    out[b] += innerprod
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_bond_amplitude, n_par, Smap, symSign, der_ids):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        derivative *= 1.0j

                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                if _np.abs(epsilon[i, w, 0]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 0]] += derivative/epsilon[i, w, 0]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    if w >= n_bond_amplitude:
                                        der *= 1.0j
                                    out[b, der_ids[i, w, 0]] += der
                        else:
                            if der_ids[i, w, 1] >= 0:
                                if _np.abs(epsilon[i, w, 1]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 1]] += derivative/epsilon[i, w, 1]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    if w >= n_bond_amplitude:
                                        der *= 1.0j
                                    out[b, der_ids[i, w, 1]] += der
        return out



class QGPSPhaseSplitSumSymReg(QGPSPhaseSplit):
    def __init__(self, hilbert, epsilon=None, n_bond_amplitude=None, n_bond_phase=None,
                 automorphisms=None, spin_flip_sym=False, cluster_ids=None):
        super().__init__(hilbert, epsilon=epsilon, n_bond_amplitude=n_bond_amplitude,
                         n_bond_phase=n_bond_phase, automorphisms=automorphisms,
                         spin_flip_sym=spin_flip_sym, cluster_ids=cluster_ids)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, n_bond_amplitude, Smap, symSign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                arg_abs = _np.complex128(0.0)
                arg_phase = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerprod *= epsilon[i, w, 0]
                        else:
                            innerprod *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        arg_phase += innerprod
                    else:
                        arg_abs += innerprod
                out[b] += _np.exp(-(arg_abs**2+1j*arg_phase**2))
            out[b] = _np.log(out[b])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_bond_amplitude, n_par, Smap, symSign, der_ids):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            value = _np.complex128(0.0)
            for t in range(Smap.shape[0]):
                argument_abs = _np.complex128(0.0)
                argument_phase = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerargument = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerargument *= epsilon[i, w, 0]
                        else:
                            innerargument *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        argument_phase += innerargument
                    else:
                        argument_abs += innerargument
                prefactor_abs = -2*argument_abs*_np.exp(-(argument_abs**2 + 1j*argument_phase**2))
                prefactor_phase = -2j*argument_phase*_np.exp(-(argument_abs**2 + 1j*argument_phase**2))

                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        prefactor = prefactor_phase
                    else:
                        prefactor = prefactor_abs
                

                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                if _np.abs(epsilon[i, w, 0]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 0]] += prefactor * derivative/epsilon[i, w, 0]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    out[b, der_ids[i, w, 0]] += prefactor * der
                        else:
                            if der_ids[i, w, 1] >= 0:
                                if _np.abs(epsilon[i, w, 1]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 1]] += prefactor * derivative/epsilon[i, w, 1]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if symSign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    out[b, der_ids[i, w, 1]] += prefactor * der
                value += _np.exp(-(argument_abs**2 + 1j*argument_phase**2))
            out[b, :] /= value
        return out
    
    def init_random_parameters(self, seed=None, sigma=0.1):
        epsilon = _np.ones(self._epsilon.shape, dtype=self._npdtype)

        if _rank == 0:
            rgen = _np.random.default_rng(seed)
            eps_shape = epsilon.shape
            epsilon[:,:,:] += rgen.normal(scale=sigma, size=eps_shape)

        if _n_nodes > 1:
            _MPI_comm.Bcast(epsilon, root=0)
            _MPI_comm.barrier()

        self._epsilon = epsilon


class QGPSPhaseSplitSumSymAltReg(QGPSPhaseSplit):
    def __init__(self, hilbert, epsilon=None, n_bond_amplitude=None, n_bond_phase=None,
                 automorphisms=None, spin_flip_sym=False, cluster_ids=None):
        super().__init__(hilbert, epsilon=epsilon, n_bond_amplitude=n_bond_amplitude,
                         n_bond_phase=n_bond_phase, automorphisms=automorphisms,
                         spin_flip_sym=spin_flip_sym, cluster_ids=cluster_ids)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, n_bond_amplitude, Smap, symSign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                arg_abs = _np.complex128(0.0)
                arg_phase = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerprod *= epsilon[i, w, 0]
                        else:
                            innerprod *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        arg_phase += innerprod**2
                    else:
                        arg_abs += innerprod**2
                out[b] += _np.exp(-(arg_abs+1j*arg_phase))
            out[b] = _np.log(out[b])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_bond_amplitude, n_par, Smap, symSign, der_ids):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            value = _np.complex128(0.0)
            for t in range(Smap.shape[0]):
                argument_abs = _np.complex128(0.0)
                argument_phase = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerargument = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            innerargument *= epsilon[i, w, 0]
                        else:
                            innerargument *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        argument_phase += innerargument**2
                    else:
                        argument_abs += innerargument**2
                prefactor_abs = -2*_np.exp(-(argument_abs + 1j*argument_phase))
                prefactor_phase = -2j*_np.exp(-(argument_abs + 1j*argument_phase))

                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        prefactor = prefactor_phase
                    else:
                        prefactor = prefactor_abs

                    for i in range(Smap.shape[1]):
                        if symSign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                if _np.abs(epsilon[i, w, 0]) > 1.e2*eps:
                                    out[b, der_ids[i, w, 0]] += prefactor * (derivative**2)/epsilon[i, w, 0]
                        else:
                            if der_ids[i, w, 1] >= 0:
                                if _np.abs(epsilon[i, w, 1]) > 1.e2*eps:
                                    out[b, der_ids[i, w, 1]] += prefactor * (derivative**2)/epsilon[i, w, 1]
                value += _np.exp(-(argument_abs + 1j*argument_phase))
            out[b, :] /= value
        return out
    
    def init_random_parameters(self, seed=None, sigma=0.1):
        epsilon = _np.ones(self._epsilon.shape, dtype=self._npdtype)

        if _rank == 0:
            rgen = _np.random.default_rng(seed)
            eps_shape = epsilon.shape
            epsilon[:,:,:] += rgen.normal(scale=sigma, size=eps_shape)

        if _n_nodes > 1:
            _MPI_comm.Bcast(epsilon, root=0)
            _MPI_comm.barrier()

        self._epsilon = epsilon



class QGPSLinExp(AbstractMachine):
    r"""
    The QGPS ansatz.
    """

    def __init__(
        self,
        hilbert,
        epsilon_lin=None,
        epsilon_exp=None,
        n_bond_lin=None,
        n_bond_exp=None,
        automorphisms=None,
        spin_flip_sym=False,
        cluster_ids=None,
        dtype=float
    ):
        n = hilbert.size

        if dtype is not float and dtype is not complex:
            raise TypeError("dtype must be either float or complex")

        self._dtype = dtype
        self._npdtype = _np.complex128 if dtype is complex else _np.float64

        if automorphisms is None:
            self._Smap_exp = _np.zeros((1, n), dtype=_np.intp)
            k = 0
            for i in range(n):
                self._Smap_exp[0, i] = k
                k += 1

        else:
            if isinstance(automorphisms, AbstractGraph):
                autom = _np.asarray(automorphisms.automorphisms())
            else:
                try:
                    autom = _np.asarray(automorphisms)
                    assert n == autom.shape[1]
                except:
                    raise RuntimeError("Cannot find a valid automorphism array.")

            self._Smap_exp = autom

        self._Smap_lin = self._Smap_exp.copy()
        
        if cluster_ids is not None:
            self._Smap_exp = self._Smap_exp[:, cluster_ids]

        self._sym_spin_flip_sign_exp = _np.ones(len(self._Smap_exp), dtype=_np.int8)
        self._sym_spin_flip_sign_lin = _np.ones(len(self._Smap_lin), dtype=_np.int8)
        if spin_flip_sym:
            self._Smap_exp = _np.append(self._Smap_exp, self._Smap_exp, axis=0)
            self._Smap_lin = _np.append(self._Smap_lin, self._Smap_lin, axis=0)
            self._sym_spin_flip_sign_exp = _np.append(self._sym_spin_flip_sign_exp,
                                                  -self._sym_spin_flip_sign_exp, axis=0)
            self._sym_spin_flip_sign_lin = _np.append(self._sym_spin_flip_sign_lin,
                                                      -self._sym_spin_flip_sign_lin, axis=0)
        if epsilon_lin is None:
            assert(n_bond_lin is not None)
            self._epsilon_lin = _np.zeros((self._Smap_lin.shape[1], n_bond_lin, 2), dtype=self._npdtype)
        else:
            self._epsilon_lin = epsilon_lin.astype(self._npdtype)

        if epsilon_exp is None:
            assert(n_bond_exp is not None)
            self._epsilon_exp = _np.zeros((self._Smap_exp.shape[1], n_bond_exp, 2), dtype=self._npdtype)
        else:
            self._epsilon_exp = epsilon_exp.astype(self._npdtype)

        self._npar = self._epsilon_lin.size + self._epsilon_exp.size
        self.value_time = 0
        self.der_time = 0

        super().__init__(hilbert, dtype=dtype)

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._npar
    
    def init_random_parameters(self, seed=None, sigma=0.1):
        epsilon_lin = _np.zeros(self._epsilon_lin.shape, dtype=self._npdtype)
        epsilon_exp = _np.zeros(self._epsilon_exp.shape, dtype=self._npdtype)

        if _rank == 0:
            rgen = _np.random.default_rng(seed)
            epsilon_lin += rgen.normal(scale=sigma, size=epsilon_lin.shape)
            epsilon_exp += rgen.normal(scale=sigma, size=epsilon_exp.shape)
            if self._dtype == complex:
                epsilon_lin += 1j*rgen.normal(scale=sigma, size=epsilon_lin.shape)
                epsilon_exp += 1j*rgen.normal(scale=sigma, size=epsilon_exp.shape)
            for i in range(epsilon_lin.shape[0]):
                for j in range(epsilon_lin.shape[1]):
                    epsilon_lin[i, j, :] /= _np.max(abs(epsilon_lin[i, j, :]))
            for i in range(epsilon_exp.shape[0]):
                for j in range(epsilon_exp.shape[1]):
                    epsilon_exp[i, j, :] /= _np.max(abs(epsilon_exp[i, j, :]))

        if _n_nodes > 1:
            _MPI_comm.Bcast(epsilon_lin, root=0)
            _MPI_comm.Bcast(epsilon_exp, root=0)
            _MPI_comm.barrier()

        self._epsilon_lin = epsilon_lin
        self._epsilon_exp = epsilon_exp

    def log_val(self, x, out=None):
        r"""Computes the logarithm of the wave function for a batch of visible
        configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. The
                 length of `out` should be `x.shape[0]`.

        Returns:
            A complex number when `x` is a vector and vector when `x` is a
            matrix.
        """
        start = time.time()
        val = self._log_val_kernel(x, out, self._epsilon_lin, self._epsilon_exp,
                                   self._Smap_lin, self._Smap_exp, self._sym_spin_flip_sign_lin,
                                   self._sym_spin_flip_sign_exp)
        self.value_time += time.time() - start
        return val

    def der_log(self, x, out=None):
        r"""Computes the gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`.
                `out` should be a matrix of shape `(v.shape[0], self.n_par)`.

        Returns:
            `out`
        """
        start = time.time()
        der = self._der_log_kernel(x, out, self._epsilon_lin, self._epsilon_exp, self._npar,
                                   self._Smap_lin, self._Smap_exp, self._sym_spin_flip_sign_lin,
                                   self._sym_spin_flip_sign_exp)
        self.der_time += time.time() - start
        return der

    @property
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        from collections import OrderedDict

        od = OrderedDict()
        if self._dtype is complex:
            od["epsilon_lin"] = self._epsilon_lin.view()
            od["epsilon_exp"] = self._epsilon_exp.view()

        else:
            self._epsilonlinc = self._epsilon_lin.astype(_np.complex128)
            self._epsilon_lin = self._epsilonlinc.real.view()
            self._epsilonexpc = self._epsilon_exp.astype(_np.complex128)
            self._epsilon_exp = self._epsilonexpc.real.view()
            od["epsilon_lin"] = self._epsilonlinc.view()
            od["epsilon_exp"] = self._epsilonexpc.view()

        return od
    
    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon_lin, epsilon_exp, Smap_lin,
                        Smap_exp, symSign_lin, symSign_exp):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            arg = _np.complex128(0.0)
            if epsilon_lin.size > 0:
                for t in range(Smap_lin.shape[0]):
                    for w in range(epsilon_lin.shape[1]):
                        innerprod = _np.complex128(1.0)
                        for i in range(Smap_lin.shape[1]):
                            if symSign_lin[t] * x[b, Smap_lin[t,i]] < 0:
                                innerprod *= epsilon_lin[i, w, 0]
                            else:
                                innerprod *= epsilon_lin[i, w, 1]
                        arg += innerprod
                out[b] += _np.log(arg)

            if epsilon_exp.size > 0:
                for t in range(Smap_exp.shape[0]):
                    for w in range(epsilon_exp.shape[1]):
                        innerprod = _np.complex128(1.0)
                        for i in range(Smap_exp.shape[1]):
                            if symSign_exp[t] * x[b, Smap_exp[t,i]] < 0:
                                innerprod *= epsilon_exp[i, w, 0]
                            else:
                                innerprod *= epsilon_exp[i, w, 1]
                        out[b] += innerprod
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon_lin, epsilon_exp, n_par, Smap_lin,
                        Smap_exp, symSign_lin, symSign_exp):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            value = _np.complex128(0.0)
            if epsilon_lin.size > 0:
                for t in range(Smap_lin.shape[0]):
                    for w in range(epsilon_lin.shape[1]):
                        derivative = _np.complex128(1.0)
                        for i in range(Smap_lin.shape[1]):
                            if symSign_lin[t] * x[b, Smap_lin[t,i]] < 0:
                                derivative *= epsilon_lin[i, w, 0]
                            else:
                                derivative *= epsilon_lin[i, w, 1]
                        for i in range(Smap_lin.shape[1]):
                            if symSign_lin[t] * x[b, Smap_lin[t,i]] < 0:
                                if _np.abs(epsilon_lin[i, w, 0]) > 1.e6*eps:
                                    out[b, 2*epsilon_lin.shape[1]*i + 2*w + 0] += derivative/epsilon_lin[i, w, 0]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap_lin.shape[1]):
                                        if j != i:
                                            if symSign_lin[t] * x[b, Smap_lin[t,j]] < 0:
                                                der *= epsilon_lin[j, w, 0]
                                            else:
                                                der *= epsilon_lin[j, w, 1]
                                    out[b, 2*epsilon_lin.shape[1]*i + 2*w + 0] += der
                            else:
                                if _np.abs(epsilon_lin[i, w, 1]) > 1.e6*eps:
                                    out[b, 2*epsilon_lin.shape[1]*i + 2*w + 1] += derivative/epsilon_lin[i, w, 1]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap_lin.shape[1]):
                                        if j != i:
                                            if symSign_lin[t] * x[b, Smap_lin[t,j]] < 0:
                                                der *= epsilon_lin[j, w, 0]
                                            else:
                                                der *= epsilon_lin[j, w, 1]
                                    out[b, 2*epsilon_lin.shape[1]*i + 2*w + 1] += der
                        value += derivative
                for i in range(epsilon_lin.size):
                    out[b, i] /= value
            if epsilon_exp.size > 0:
                for t in range(Smap_exp.shape[0]):
                    for w in range(epsilon_exp.shape[1]):
                        derivative = _np.complex128(1.0)
                        for i in range(Smap_exp.shape[1]):
                            if symSign_exp[t] * x[b, Smap_exp[t,i]] < 0:
                                derivative *= epsilon_exp[i, w, 0]
                            else:
                                derivative *= epsilon_exp[i, w, 1]
                        for i in range(Smap_exp.shape[1]):
                            if symSign_exp[t] * x[b, Smap_exp[t,i]] < 0:
                                if _np.abs(epsilon_exp[i, w, 0]) > 1.e6*eps:
                                    out[b, epsilon_lin.size + 2*epsilon_exp.shape[1]*i + 2*w + 0] += derivative/epsilon_exp[i, w, 0]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap_exp.shape[1]):
                                        if j != i:
                                            if symSign_exp[t] * x[b, Smap_exp[t,j]] < 0:
                                                der *= epsilon_exp[j, w, 0]
                                            else:
                                                der *= epsilon_exp[j, w, 1]
                                    out[b, epsilon_lin.size + 2*epsilon_exp.shape[1]*i + 2*w + 0] += der
                            else:
                                if _np.abs(epsilon_exp[i, w, 1]) > 1.e6*eps:
                                    out[b, epsilon_lin.size + 2*epsilon_exp.shape[1]*i + 2*w + 1] += derivative/epsilon_exp[i, w, 1]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap_exp.shape[1]):
                                        if j != i:
                                            if symSign_exp[t] * x[b, Smap_exp[t,j]] < 0:
                                                der *= epsilon_exp[j, w, 0]
                                            else:
                                                der *= epsilon_exp[j, w, 1]
                                    out[b, epsilon_lin.size + 2*epsilon_exp.shape[1]*i + 2*w + 1] += der
        return out
