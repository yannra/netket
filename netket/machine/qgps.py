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

        self._exp_kern_representation = True

        self._site_product = _np.zeros((self._epsilon.shape[1], self._Smap.shape[0]), dtype=_np.complex128)
        self._ref_conf = None
        self._Smap_inverse = -1 * _np.ones((self._Smap.shape[0], self._Smap.max()+1), dtype=_np.int)

        self._fast_update = True

        for i in range(self._Smap.shape[0]):
            for j in range(self._Smap.shape[1]):
                self._Smap_inverse[i, self._Smap[i,j]] = j

        super().__init__(hilbert, dtype=dtype)
    
    def reset(self):
        self._ref_conf = None

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._opt_params.size
    
    def init_random_parameters(self, seed=None, sigma=0.1, start_from_uniform=True):
        if self._exp_kern_representation:
            epsilon = _np.zeros(self._epsilon.shape, dtype=self._npdtype)

            if _rank == 0:
                rgen = _np.random.default_rng(seed)
                epsilon += rgen.normal(scale=sigma, size=epsilon.shape)
                if self._dtype == complex:
                    epsilon += 1j*rgen.normal(scale=sigma, size=epsilon.shape)
                if start_from_uniform:
                    epsilon[1:,:,:] = 0.
        else:
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
        if self._ref_conf is None or not self._fast_update:
            if len(x.shape) > 1:
                self._ref_conf = x[0,:].copy()
            else:
                self._ref_conf = x.copy()
            if self._exp_kern_representation:
                self._compute_site_prod_exp_form(self._ref_conf, self._site_product, self._epsilon, self._Smap, self._Smap_inverse, self._sym_spin_flip_sign)
            else:
                self._compute_site_prod_std_form(self._ref_conf, self._site_product, self._epsilon, self._Smap, self._Smap_inverse, self._sym_spin_flip_sign)
        if self._exp_kern_representation:
            val = self._log_val_kernel_exp_form(x, out, self._ref_conf, self._site_product, self._epsilon, self._Smap, self._Smap_inverse, self._sym_spin_flip_sign)
        else:
            val = self._log_val_kernel_std_form(x, out, self._ref_conf, self._site_product, self._epsilon, self._Smap, self._Smap_inverse, self._sym_spin_flip_sign)
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
        if self._ref_conf is None or not self._fast_update:
            if len(x.shape) > 1:
                self._ref_conf = x[0,:].copy()
            else:
                self._ref_conf = x.copy()
            if self._exp_kern_representation:
                self._compute_site_prod_exp_form(self._ref_conf, self._site_product, self._epsilon, self._Smap, self._Smap_inverse, self._sym_spin_flip_sign)
            else:
                self._compute_site_prod_std_form(self._ref_conf, self._site_product, self._epsilon, self._Smap, self._Smap_inverse, self._sym_spin_flip_sign)
        if self._exp_kern_representation:
            der = self._der_log_kernel_exp_form(x, out,  self._ref_conf, self._site_product, self._epsilon, self._npar, self._Smap, self._Smap_inverse, self._sym_spin_flip_sign, self._der_ids)
        else:
            der = self._der_log_kernel_std_form(x, out,  self._ref_conf, self._site_product, self._epsilon, self._npar, self._Smap, self._Smap_inverse, self._sym_spin_flip_sign, self._der_ids)
        self.der_time += time.time() - start
        return der

    @staticmethod
    @jit(nopython=True)
    def _compute_site_prod_std_form(ref_conf, site_product, epsilon, Smap, Smap_inverse, sym_spin_flip_sign):
        for t in range(Smap.shape[0]):
            for w in range(epsilon.shape[1]):
                innerprod = _np.complex128(1.0)
                for i in range(Smap.shape[1]):
                    if sym_spin_flip_sign[t] * ref_conf[Smap[t,i]] < 0:
                        innerprod *= epsilon[i, w, 0]
                    else:
                        innerprod *= epsilon[i, w, 1]
                site_product[w, t] = innerprod
    
    @staticmethod
    @jit(nopython=True)
    def _compute_site_prod_exp_form(ref_conf, site_product, epsilon, Smap, Smap_inverse, sym_spin_flip_sign):
        for t in range(Smap.shape[0]):
            for w in range(epsilon.shape[1]):
                innerprod = _np.complex128(0.0)
                for i in range(Smap.shape[1]):
                    if sym_spin_flip_sign[t] * ref_conf[Smap[t,i]] < 0:
                        innerprod += epsilon[i, w, 0]
                    else:
                        innerprod += epsilon[i, w, 1]
                site_product[w, t] = innerprod

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
        self.reset()
    

class QGPSSumSym(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, cluster_ids=None, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         cluster_ids=cluster_ids, dtype=dtype)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel_exp_form(x, out, ref_conf, site_product, epsilon, Smap, Smap_inverse, sym_spin_flip_sign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0

            # update site product
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for t in range(Smap.shape[0]):
                        i = Smap_inverse[t, pos]
                        if i != -1:
                            for w in range(epsilon.shape[1]):
                                if sym_spin_flip_sign[t] * x[b, pos] < 0:
                                    site_product[w, t] += (epsilon[i, w, 0] - epsilon[i, w, 1])
                                else:
                                    site_product[w, t] += (epsilon[i, w, 1] - epsilon[i, w, 0])
                ref_conf[pos] = x[b, pos]

            for t in range(Smap.shape[0]):
                arg = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    arg += _np.exp(site_product[w, t])
                out[b] += _np.exp(arg)
            out[b] = _np.log(out[b])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel_exp_form(x, out, ref_conf, site_product, epsilon, npar, Smap, Smap_inverse, sym_spin_flip_sign, der_ids):
        batch_size = x.shape[0]

        if out is None:
            out = _np.empty((batch_size, npar), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            #update site product
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for t in range(Smap.shape[0]):
                        i = Smap_inverse[t, pos]
                        if i != -1:
                            for w in range(epsilon.shape[1]):
                                if sym_spin_flip_sign[t] * x[b, pos] < 0:
                                    site_product[w, t] += (epsilon[i, w, 0]-epsilon[i, w, 1])
                                else:
                                    site_product[w, t] += (epsilon[i, w, 1]-epsilon[i, w, 0])
                ref_conf[pos] = x[b, pos]

            value = _np.complex128(0.0)
            for t in range(Smap.shape[0]):
                argument = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    argument += _np.exp(site_product[w,t])
                prefactor = _np.exp(argument)

                for w in range(epsilon.shape[1]):
                    derivative = _np.exp(site_product[w,t])
                    for i in range(Smap.shape[1]):
                        if sym_spin_flip_sign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                out[b, der_ids[i, w, 0]] += prefactor * derivative
                        else:
                            if der_ids[i, w, 1] >= 0:
                                out[b, der_ids[i, w, 1]] += prefactor * derivative
                value += prefactor
            out[b, :] /= value
        return out
    
    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel_std_form(x, out, ref_conf, site_product, epsilon, Smap, Smap_inverse, sym_spin_flip_sign):
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0

            # update site product
            recompute = False
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for t in range(Smap.shape[0]):
                        if recompute:
                            break
                        i = Smap_inverse[t, pos]
                        if i != -1:
                            for w in range(epsilon.shape[1]):
                                # TODO: careful about 0 values -> better implementation
                                if sym_spin_flip_sign[t] * x[b, pos] < 0:
                                    if abs(epsilon[i, w, 1]) > 1.e4 * eps:
                                        site_product[w, t] *= (epsilon[i, w, 0]/epsilon[i, w, 1])
                                    else:
                                        recompute = True
                                else:
                                    if abs(epsilon[i, w, 0]) > 1.e4 * eps:
                                        site_product[w, t] *= (epsilon[i, w, 1]/epsilon[i, w, 0])
                                    else:
                                        recompute = True
                                if recompute:
                                    break
                ref_conf[pos] = x[b, pos]
            if recompute:
                for t in range(Smap.shape[0]):
                    for w in range(epsilon.shape[1]):
                        site_product[w, t] = _np.complex128(1.0)
                        for i in range(Smap.shape[1]):
                            if sym_spin_flip_sign[t] * ref_conf[Smap[t,i]] < 0:
                                site_product[w, t] *= epsilon[i, w, 0]
                            else:
                                site_product[w, t] *= epsilon[i, w, 1]

            for t in range(Smap.shape[0]):
                arg = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    arg += site_product[w, t]
                out[b] += _np.exp(arg)
            out[b] = _np.log(out[b])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel_std_form(x, out, ref_conf, site_product, epsilon, npar, Smap, Smap_inverse, sym_spin_flip_sign, der_ids):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, npar), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(x.shape[0]):
            out[b] = 0.0

            # update site product
            recompute = False
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for t in range(Smap.shape[0]):
                        if recompute:
                            break
                        i = Smap_inverse[t, pos]
                        if i != -1:
                            for w in range(epsilon.shape[1]):
                                # TODO: careful about 0 values -> better implementation
                                if sym_spin_flip_sign[t] * x[b, pos] < 0:
                                    if abs(epsilon[i, w, 1]) > 1.e4 * eps:
                                        site_product[w, t] *= (epsilon[i, w, 0]/epsilon[i, w, 1])
                                    else:
                                        recompute = True
                                else:
                                    if abs(epsilon[i, w, 0]) > 1.e4 * eps:
                                        site_product[w, t] *= (epsilon[i, w, 1]/epsilon[i, w, 0])
                                    else:
                                        recompute = True
                                if recompute:
                                    break
                ref_conf[pos] = x[b, pos]
            if recompute:
                for t in range(Smap.shape[0]):
                    for w in range(epsilon.shape[1]):
                        site_product[w, t] = _np.complex128(1.0)
                        for i in range(Smap.shape[1]):
                            if sym_spin_flip_sign[t] * ref_conf[Smap[t,i]] < 0:
                                site_product[w, t] *= epsilon[i, w, 0]
                            else:
                                site_product[w, t] *= epsilon[i, w, 1]

            value = _np.complex128(0.0)
            for t in range(Smap.shape[0]):
                argument = _np.complex128(0.0)
                for w in range(epsilon.shape[1]):
                    argument += site_product[w,t]
                prefactor = _np.exp(argument)

                for w in range(epsilon.shape[1]):
                    derivative = site_product[w,t]
                    for i in range(Smap.shape[1]):
                        if sym_spin_flip_sign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                if _np.abs(epsilon[i, w, 0]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 0]] += prefactor * derivative/epsilon[i, w, 0]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if sym_spin_flip_sign[t] * x[b, Smap[t,j]] < 0:
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
                                            if sym_spin_flip_sign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    out[b, der_ids[i, w, 1]] += prefactor * der
                value += prefactor
            out[b, :] /= value
        return out


class QGPSBasisSym(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         cluster_ids=None, dtype=dtype)
        self._workspace = None
        self._site_product = _np.zeros(self._epsilon.shape[1], dtype=_np.complex128)

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
        if self._workspace is None:
            self._workspace = x[0,:].copy()

        self._set_master_conf(x, self._workspace, self._Smap, self._sym_spin_flip_sign)
        
        if self._ref_conf is None or not self._fast_update:
            if len(x.shape) > 1:
                self._ref_conf = x[0,:].copy()
            else:
                self._ref_conf = x.copy()
            if self._exp_kern_representation:
                self._compute_site_prod_exp_form(self._ref_conf, self._site_product, self._epsilon)
            else:
                self._compute_site_prod_std_form(self._ref_conf, self._site_product, self._epsilon)
        if self._exp_kern_representation:
            val = self._log_val_kernel_exp_form(x, out, self._ref_conf, self._site_product, self._epsilon)
        else:
            val = self._log_val_kernel_std_form(x, out, self._ref_conf, self._site_product, self._epsilon)
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

        if self._workspace is None:
            self._workspace = x[0,:].copy()

        self._set_master_conf(x, self._workspace, self._Smap, self._sym_spin_flip_sign)
        
        if self._ref_conf is None or not self._fast_update:
            if len(x.shape) > 1:
                self._ref_conf = x[0,:].copy()
            else:
                self._ref_conf = x.copy()
            if self._exp_kern_representation:
                self._compute_site_prod_exp_form(self._ref_conf, self._site_product, self._epsilon)
            else:
                self._compute_site_prod_std_form(self._ref_conf, self._site_product, self._epsilon)
        if self._exp_kern_representation:
            der = self._der_log_kernel_exp_form(x, out,  self._ref_conf, self._site_product, self._epsilon, self._npar, self._der_ids)
        else:
            der = self._der_log_kernel_std_form(x, out,  self._ref_conf, self._site_product, self._epsilon, self._npar, self._der_ids)
        return der

    @staticmethod
    @jit(nopython=True)
    def _set_master_conf(x, workspace, Smap, sym_spin_flip_sign):
        limit = x.shape[1] - 1
        for b in range(x.shape[0]):
            t = 0
            for i in range(workspace.shape[0]):
                workspace[i] = sym_spin_flip_sign[t] * x[b, Smap[t,i]]
            for t in range(1, Smap.shape[0]):
                pos = 0
                while sym_spin_flip_sign[t] * x[b, Smap[t,pos]] == workspace[pos] and pos < limit:
                    pos += 1
                if sym_spin_flip_sign[t] * x[b, Smap[t,pos]] > workspace[pos]:
                    for i in range(workspace.shape[0]):
                        workspace[i] = sym_spin_flip_sign[t] * x[b, Smap[t,i]]

            for i in range(workspace.shape[0]):
                x[b, i] = workspace[i]

    @staticmethod
    @jit(nopython=True)
    def _compute_site_prod_std_form(ref_conf, site_product, epsilon):
        for w in range(epsilon.shape[1]):
            innerprod = _np.complex128(1.0)
            for i in range(ref_conf.shape[0]):
                if ref_conf[i] < 0:
                    innerprod *= epsilon[i, w, 0]
                else:
                    innerprod *= epsilon[i, w, 1]
            site_product[w] = innerprod
    
    @staticmethod
    @jit(nopython=True)
    def _compute_site_prod_exp_form(ref_conf, site_product, epsilon):
        for w in range(epsilon.shape[1]):
            innerprod = _np.complex128(0.0)
            for i in range(ref_conf.shape[0]):
                if ref_conf[i] < 0:
                    innerprod += epsilon[i, w, 0]
                else:
                    innerprod += epsilon[i, w, 1]
            site_product[w] = innerprod

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel_exp_form(x, out, ref_conf, site_product, epsilon):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0

            # update site product
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for w in range(epsilon.shape[1]):
                        if x[b, pos] < 0:
                            site_product[w] += (epsilon[pos, w, 0] - epsilon[pos, w, 1])
                        else:
                            site_product[w] += (epsilon[pos, w, 1] - epsilon[pos, w, 0])
                ref_conf[pos] = x[b, pos]

            out[b] = _np.complex128(0.0)
            for w in range(epsilon.shape[1]):
                out[b] += _np.exp(site_product[w])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel_exp_form(x, out, ref_conf, site_product, epsilon, npar, der_ids):
        batch_size = x.shape[0]

        if out is None:
            out = _np.empty((batch_size, npar), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            # update site product
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for w in range(epsilon.shape[1]):
                        if x[b, pos] < 0:
                            site_product[w] += (epsilon[pos, w, 0] - epsilon[pos, w, 1])
                        else:
                            site_product[w] += (epsilon[pos, w, 1] - epsilon[pos, w, 0])
                ref_conf[pos] = x[b, pos]

            for w in range(epsilon.shape[1]):
                derivative = _np.exp(site_product[w])
                for i in range(x.shape[1]):
                    if x[b, i] < 0:
                        if der_ids[i, w, 0] >= 0:
                            out[b, der_ids[i, w, 0]] += derivative
                    else:
                        if der_ids[i, w, 1] >= 0:
                            out[b, der_ids[i, w, 1]] += derivative
        return out

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel_std_form(x, out, ref_conf, site_product, epsilon):
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0

            # update site product
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    i = pos
                    for w in range(epsilon.shape[1]):
                        if x[b, pos] < 0:
                            if abs(epsilon[pos, w, 1]) > 1.e4 * eps:
                                site_product[w] *= (epsilon[pos, w, 0]/epsilon[pos, w, 1])
                            else:
                                recompute = True
                        else:
                            if abs(epsilon[pos, w, 0]) > 1.e4 * eps:
                                site_product[w] *= (epsilon[pos, w, 1]/epsilon[pos, w, 0])
                            else:
                                recompute = True
                        if recompute:
                            break
                ref_conf[pos] = x[b, pos]
            if recompute:
                for w in range(epsilon.shape[1]):
                    site_product[w] = _np.complex128(1.0)
                    for i in range(ref_conf.shape[0]):
                        if ref_conf[i] < 0:
                            site_product[w] *= epsilon[i, w, 0]
                        else:
                            site_product[w] *= epsilon[i, w, 1]

            out[b] = _np.complex128(0.0)
            for w in range(epsilon.shape[1]):
                out[b] += site_product[w]
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel_std_form(x, out, ref_conf, site_product, epsilon, npar, der_ids):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, npar), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(x.shape[0]):
            out[b] = 0.0

            # update site product
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for w in range(epsilon.shape[1]):
                        if x[b, pos] < 0:
                            if abs(epsilon[pos, w, 1]) > 1.e4 * eps:
                                site_product[w] *= (epsilon[pos, w, 0]/epsilon[pos, w, 1])
                            else:
                                recompute = True
                        else:
                            if abs(epsilon[pos, w, 0]) > 1.e4 * eps:
                                site_product[w] *= (epsilon[pos, w, 1]/epsilon[pos, w, 0])
                            else:
                                recompute = True
                        if recompute:
                            break
                ref_conf[pos] = x[b, pos]
            if recompute:
                for w in range(epsilon.shape[1]):
                    site_product[w] = _np.complex128(1.0)
                    for i in range(ref_conf.shape[0]):
                        if ref_conf[i] < 0:
                            site_product[w] *= epsilon[i, w, 0]
                        else:
                            site_product[w] *= epsilon[i, w, 1]

            for w in range(epsilon.shape[1]):
                derivative = site_product[w]
                for i in range(x.shape[1]):
                    if x[b, i] < 0:
                        if der_ids[i, w, 0] >= 0:
                            if _np.abs(epsilon[i, w, 0]) > 1.e6*eps:
                                out[b, der_ids[i, w, 0]] += derivative/epsilon[i, w, 0]
                            else:
                                der = _np.complex128(1.0)
                                for j in range(x.shape[1]):
                                    if j != i:
                                        if x[b, j] < 0:
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
                                for j in range(x.shape[1]):
                                    if j != i:
                                        if x[b, j] < 0:
                                            der *= epsilon[j, w, 0]
                                        else:
                                            der *= epsilon[j, w, 1]
                                out[b, der_ids[i, w, 1]] += der
        return out



class QGPSProdSym(QGPS):
    def __init__(self, hilbert, epsilon=None, n_bond=None, automorphisms=None,
                 spin_flip_sym=False, cluster_ids=None, dtype=complex):
        super().__init__(hilbert, epsilon=epsilon, n_bond=n_bond,
                         automorphisms=automorphisms, spin_flip_sym=spin_flip_sym,
                         cluster_ids=cluster_ids, dtype=dtype)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel_exp_form(x, out, ref_conf, site_product, epsilon, Smap, Smap_inverse, sym_spin_flip_sign):
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0

            # update site product
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for t in range(Smap.shape[0]):
                        i = Smap_inverse[t, pos]
                        if i != -1:
                            for w in range(epsilon.shape[1]):
                                if sym_spin_flip_sign[t] * x[b, pos] < 0:
                                    site_product[w, t] += (epsilon[i, w, 0] - epsilon[i, w, 1])
                                else:
                                    site_product[w, t] += (epsilon[i, w, 1] - epsilon[i, w, 0])
                ref_conf[pos] = x[b, pos]

            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    out[b] += _np.exp(site_product[w, t])
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel_exp_form(x, out, ref_conf, site_product, epsilon, npar, Smap, Smap_inverse, sym_spin_flip_sign, der_ids):
        batch_size = x.shape[0]
        if out is None:
            out = _np.empty((batch_size, npar), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            # update site product
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for t in range(Smap.shape[0]):
                        i = Smap_inverse[t, pos]
                        if i != -1:
                            for w in range(epsilon.shape[1]):
                                if sym_spin_flip_sign[t] * x[b, pos] < 0:
                                    site_product[w, t] += (epsilon[i, w, 0]-epsilon[i, w, 1])
                                else:
                                    site_product[w, t] += (epsilon[i, w, 1]-epsilon[i, w, 0])
                ref_conf[pos] = x[b, pos]

            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    derivative = _np.exp(site_product[w,t])
                    for i in range(Smap.shape[1]):
                        if sym_spin_flip_sign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                out[b, der_ids[i, w, 0]] += derivative
                        else:
                            if der_ids[i, w, 1] >= 0:
                                out[b, der_ids[i, w, 1]] += derivative
        return out
    
    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel_std_form(x, out, ref_conf, site_product, epsilon, Smap, Smap_inverse, sym_spin_flip_sign):
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        for b in range(x.shape[0]):
            out[b] = 0.0

            # update site product
            recompute = False
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for t in range(Smap.shape[0]):
                        if recompute:
                            break
                        i = Smap_inverse[t, pos]
                        if i != -1:
                            for w in range(epsilon.shape[1]):
                                # TODO: careful about 0 values -> better implementation
                                if sym_spin_flip_sign[t] * x[b, pos] < 0:
                                    if abs(epsilon[i, w, 1]) > 1.e4 * eps:
                                        site_product[w, t] *= (epsilon[i, w, 0]/epsilon[i, w, 1])
                                    else:
                                        recompute = True
                                else:
                                    if abs(epsilon[i, w, 0]) > 1.e4 * eps:
                                        site_product[w, t] *= (epsilon[i, w, 1]/epsilon[i, w, 0])
                                    else:
                                        recompute = True
                                if recompute:
                                    break
                ref_conf[pos] = x[b, pos]
            if recompute:
                for t in range(Smap.shape[0]):
                    for w in range(epsilon.shape[1]):
                        site_product[w, t] = _np.complex128(1.0)
                        for i in range(Smap.shape[1]):
                            if sym_spin_flip_sign[t] * ref_conf[Smap[t,i]] < 0:
                                site_product[w, t] *= epsilon[i, w, 0]
                            else:
                                site_product[w, t] *= epsilon[i, w, 1]

            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    out[b] += site_product[w, t]
        return out

    @staticmethod
    @jit(nopython=True)
    def _der_log_kernel_std_form(x, out, ref_conf, site_product, epsilon, npar, Smap, Smap_inverse, sym_spin_flip_sign, der_ids):
        batch_size = x.shape[0]
        eps = _np.finfo(_np.double).eps

        if out is None:
            out = _np.empty((batch_size, npar), dtype=_np.complex128)

        out.fill(0.0)

        for b in range(batch_size):
            # update site product
            recompute = False
            for pos in range(x.shape[1]):
                if x[b, pos] != ref_conf[pos]:
                    for t in range(Smap.shape[0]):
                        if recompute:
                            break
                        i = Smap_inverse[t, pos]
                        if i != -1:
                            for w in range(epsilon.shape[1]):
                                # TODO: careful about 0 values -> better implementation
                                if sym_spin_flip_sign[t] * x[b, pos] < 0:
                                    if abs(epsilon[i, w, 1]) > 1.e4 * eps:
                                        site_product[w, t] *= (epsilon[i, w, 0]/epsilon[i, w, 1])
                                    else:
                                        recompute = True
                                else:
                                    if abs(epsilon[i, w, 0]) > 1.e4 * eps:
                                        site_product[w, t] *= (epsilon[i, w, 1]/epsilon[i, w, 0])
                                    else:
                                        recompute = True
                                if recompute:
                                    break
                ref_conf[pos] = x[b, pos]
            if recompute:
                for t in range(Smap.shape[0]):
                    for w in range(epsilon.shape[1]):
                        site_product[w, t] = _np.complex128(1.0)
                        for i in range(Smap.shape[1]):
                            if sym_spin_flip_sign[t] * ref_conf[Smap[t,i]] < 0:
                                site_product[w, t] *= epsilon[i, w, 0]
                            else:
                                site_product[w, t] *= epsilon[i, w, 1]

            for t in range(Smap.shape[0]):
                for w in range(epsilon.shape[1]):
                    derivative = site_product[w,t]
                    for i in range(Smap.shape[1]):
                        if sym_spin_flip_sign[t] * x[b, Smap[t,i]] < 0:
                            if der_ids[i, w, 0] >= 0:
                                if _np.abs(epsilon[i, w, 0]) > 1.e6*eps:
                                    out[b, der_ids[i, w, 0]] += derivative/epsilon[i, w, 0]
                                else:
                                    der = _np.complex128(1.0)
                                    for j in range(Smap.shape[1]):
                                        if j != i:
                                            if sym_spin_flip_sign[t] * x[b, Smap[t,j]] < 0:
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
                                            if sym_spin_flip_sign[t] * x[b, Smap[t,j]] < 0:
                                                der *= epsilon[j, w, 0]
                                            else:
                                                der *= epsilon[j, w, 1]
                                    out[b, der_ids[i, w, 1]] += der
        return out


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
        epsilon_lin = _np.ones(self._epsilon_lin.shape, dtype=self._npdtype)
        epsilon_exp = _np.ones(self._epsilon_exp.shape, dtype=self._npdtype)

        if _rank == 0:
            rgen = _np.random.default_rng(seed)
            epsilon_lin += rgen.normal(scale=sigma, size=epsilon_lin.shape)
            epsilon_exp += rgen.normal(scale=sigma, size=epsilon_exp.shape)
            if self._dtype == complex:
                epsilon_lin += 1j*rgen.normal(scale=sigma, size=epsilon_lin.shape)
                epsilon_exp += 1j*rgen.normal(scale=sigma, size=epsilon_exp.shape)

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
