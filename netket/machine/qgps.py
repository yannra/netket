from .abstract_machine import AbstractMachine
from netket.graph import AbstractGraph
import numpy as _np
from numba import jit
import time


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
        dtype=complex,
        outdtype=None
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
        
        if spin_flip_sym:
            self._Smap = _np.append(self._Smap, -self._Smap, axis=0)

        if epsilon is None:
            assert(n_bond is not None)
            self._epsilon = _np.zeros((n, n_bond, 2), dtype=self._npdtype)
        else:
            self._epsilon = epsilon.astype(self._npdtype)

        self._npar = self._epsilon.size
        self.value_time = 0
        self.der_time = 0

        super().__init__(hilbert, dtype=dtype, outdtype=outdtype)

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._npar

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
        val = self._log_val_kernel(x, out, self._epsilon, self._Smap)
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
        der = self._der_log_kernel(x, out, self._epsilon, self._npar, self._Smap)
        self.der_time += time.time() - start
        return der

    @property
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        from collections import OrderedDict

        od = OrderedDict()
        if self._dtype is complex:
            od["epsilon"] = self._epsilon.view()

        else:
            self._epsilonc = self._epsilon.astype(_np.complex128)
            self._epsilon = self._epsilonc.real.view()
            od["epsilon"] = self._epsilonc.view()

        return od
    
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

class QGPSSumSym(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         dtype=dtype)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, Smap):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                arg = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(x.shape[1]):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            innerprod *= epsilon[i, w, 0]
                        else:
                            innerprod *= epsilon[i, w, 1]
                    arg += innerprod
                out[b] += _np.exp(arg)
            out[b] = _np.log(out[b])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_par, Smap):
        batch_size = x.shape[0]

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        n = x.shape[1]

        for b in range(batch_size):
            value = _np.complex128(0.0)
            for t in range(Smap.shape[0]):
                argument = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerargument = _np.complex128(1.0)
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            innerargument *= epsilon[i, w, 0]
                        else:
                            innerargument *= epsilon[i, w, 1]
                    argument += innerargument
                prefactor = _np.exp(argument)

                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            out[b, 2*epsilon.shape[1]*i + 2*w + 0] += prefactor * derivative/epsilon[i, w, 0]
                        else:
                            out[b, 2*epsilon.shape[1]*i + 2*w + 1] += prefactor * derivative/epsilon[i, w, 1]
                value += prefactor
            out[b, :] /= value
        return out


class QGPSProdSym(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         dtype=dtype)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, Smap):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(x.shape[1]):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            innerprod *= epsilon[i, w, 0]
                        else:
                            innerprod *= epsilon[i, w, 1]
                    out[b] += innerprod
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_par, Smap):
        batch_size = x.shape[0]

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        n = x.shape[1]

        for b in range(batch_size):
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            out[b, 2*epsilon.shape[1]*i + 2*w + 0] += derivative/epsilon[i, w, 0]
                        else:
                            out[b, 2*epsilon.shape[1]*i + 2*w + 1] += derivative/epsilon[i, w, 1]
        return out

class QGPSPhaseSplit(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond_amplitude=None, n_bond_phase=None,
                 automorphisms=None, spin_flip_sym=False):
        self.n_bond_amplitude = n_bond_amplitude

        if n_bond_phase is not None and n_bond_amplitude is not None and epsilon is not None:
            n_bond_total = n_bond_phase + n_bond_amplitude
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
                         dtype=float, outdtype=complex)

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
        val = self._log_val_kernel(x, out, self._epsilon, self.n_bond_amplitude, self._Smap)
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
                                   self._npar, self._Smap)
        self.der_time += time.time() - start
        return der


class QGPSPhaseSplitSumSym(QGPSPhaseSplit):
    def __init__(self, hilbert, epsilon=None, n_bond_amplitude=None, n_bond_phase=None,
                 automorphisms=None, spin_flip_sym=False):
        super().__init__(hilbert, epsilon=epsilon, n_bond_amplitude=n_bond_amplitude,
                         n_bond_phase=n_bond_phase, automorphisms=automorphisms,
                         spin_flip_sym=spin_flip_sym)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, n_bond_amplitude, Smap):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                arg = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(x.shape[1]):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
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
    def _der_log_kernel(x, out, epsilon, n_bond_amplitude, n_par, Smap):
        batch_size = x.shape[0]

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        n = x.shape[1]

        for b in range(batch_size):
            value = _np.complex128(0.0)
            for t in range(Smap.shape[0]):
                argument = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    innerargument = _np.complex128(1.0)
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            innerargument *= epsilon[i, w, 0]
                        else:
                            innerargument *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        innerargument *= 1.0j
                    argument += innerargument
                prefactor = _np.exp(argument)

                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        derivative *= 1.0j
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            out[b, 2*epsilon.shape[1]*i + 2*w + 0] += prefactor * derivative/epsilon[i, w, 0]
                        else:
                            out[b, 2*epsilon.shape[1]*i + 2*w + 1] += prefactor * derivative/epsilon[i, w, 1]
                value += prefactor
            out[b, :] /= value
        return out


class QGPSPhaseSplitProdSym(QGPSPhaseSplit):
    def __init__(self, hilbert, epsilon=None, n_bond_amplitude=None, n_bond_phase=None,
                 automorphisms=None, spin_flip_sym=False):
        super().__init__(hilbert, epsilon=epsilon, n_bond_amplitude=n_bond_amplitude,
                         n_bond_phase=n_bond_phase, automorphisms=automorphisms,
                         spin_flip_sym=spin_flip_sym)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, epsilon, n_bond_amplitude, Smap):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    innerprod = _np.complex128(1.0)
                    for i in range(x.shape[1]):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            innerprod *= epsilon[i, w, 0]
                        else:
                            innerprod *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        innerprod *= 1.0j
                    out[b] += innerprod
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel(x, out, epsilon, n_bond_amplitude, n_par, Smap):
        batch_size = x.shape[0]

        if out is None:
            out = _np.empty((batch_size, n_par), dtype=_np.complex128)

        out.fill(0.0)

        n = x.shape[1]

        for b in range(batch_size):
            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    derivative = _np.complex128(1.0)
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            derivative *= epsilon[i, w, 0]
                        else:
                            derivative *= epsilon[i, w, 1]
                    if w >= n_bond_amplitude:
                        derivative *= 1.0j
                    for i in range(n):
                        if _np.sign(Smap[t,i]) * x[b, _np.abs(Smap[t,i])] > 0:
                            out[b, 2*epsilon.shape[1]*i + 2*w + 0] += derivative/epsilon[i, w, 0]
                        else:
                            out[b, 2*epsilon.shape[1]*i + 2*w + 1] += derivative/epsilon[i, w, 1]
        return out