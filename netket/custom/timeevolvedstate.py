import netket.machine as machine
import numpy as np
from numba import jit

from netket.utils import (
    MPI_comm as _MPI_comm,
    n_nodes as _n_nodes,
    node_number as _rank
)

from netket.stats import (
    statistics as _statistics,
    mean as _mean,
    sum_inplace as _sum_inplace,
)

class TimeEvolvedState(machine.AbstractMachine):
    def __init__(self, machine, hamiltonian, beta=1., order=1):
        super().__init__(machine.hilbert, dtype=machine.dtype)
        self.machine = machine
        self.ha = hamiltonian
        self.beta = beta
        self.order = order

    def log_val(self, x, out=None):
        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        assert (
            x.shape[1] == self.ha.hilbert.size
        ), "samples has wrong shape: {}; expected (?, {})".format(v.shape, self.ha.hilbert.size)

        if out is None:
            out = np.zeros(x.shape[0], dtype=np.complex128)
        else:
            out.fill(0.0)

        sections = [np.array(range(1,x.shape[0]+1), dtype=np.int32)]
        x_primes = np.asarray(x)
        log_val_primes = [self.machine.log_val(x_primes)]
        mels = [np.ones(x.shape[0], dtype=np.float)]

        for j in range(self.order):
            sections.append(np.empty(x_primes.shape[0], dtype=np.int32))
            x_primes, mels_new = self.ha.get_conn_flattened(x_primes, sections[-1])
            mels.append(mels_new)
            log_val_primes.append(self.machine.log_val(x_primes))
 
        for j in range(self.order+1):
            if j != 0:
                arg = np.ones(log_val_primes[j-1].size, dtype=np.complex128)
                arg = self.contract_inner_sum((np.exp(log_val_primes[j])*mels[j]), sections[j], arg)

            else:
                arg = np.exp(log_val_primes[j])

            for k in range(j-1, 0, -1):
                arg_new = np.ones(log_val_primes[k-1].size, dtype=np.complex128)
                arg_new = self.contract_inner_sum((arg * mels[k]), sections[k], arg_new)
                arg = arg_new

            out += ((-self.beta)**j/np.math.factorial(j)) * arg
        out = np.log(out)
        return out

    def der_log(self, x, out=None):
        return out

    def get_h_vals(self, samples):
        log_vals = self.machine.log_val(samples)

        sections = np.empty(samples.shape[0], dtype=np.int32)
        v_primes, mels = self.ha.get_conn_flattened(np.asarray(samples), sections)

        sections2 = np.empty(v_primes.shape[0], dtype=np.int32)
        v_primes2, mels2 = self.ha.get_conn_flattened(v_primes, sections2)

        log_val_primes = self.machine.log_val(v_primes)
        log_val_primes2 = self.machine.log_val(v_primes2)

        arg = np.ones(log_val_primes.size, dtype=np.complex128)
        arg = self.contract_inner_sum((np.exp(log_val_primes2)*mels2), sections2, arg)

        h2_exp = np.ones(samples.shape[0], dtype=np.complex128)
        h2_exp = self.contract_inner_sum((arg * mels), sections, h2_exp)

        h_exp = np.ones(samples.shape[0], dtype=np.complex128)
        h_exp = self.contract_inner_sum((np.exp(log_val_primes)*mels), sections, h_exp)

        h2_exp /= np.exp(log_vals)
        h_exp /= np.exp(log_vals)

        h1 = _mean(h_exp.conj())
        h2a = _mean(h_exp.conj() * h_exp)
        h2b = _mean(h2_exp.conj())
        h3 = _mean(h_exp.conj()*h2_exp)

        return (h1, h2a, h2b, h3)

    def estimate_energy(self, samples):
        assert(self.order == 1)
        h1, h2a, h2b, h3 = self.get_h_vals(samples)

        E = h1 - self.beta *(h2a + h2b) + self.beta**2 * h3
        E /= (1 - self.beta * h1 + self.beta**2 * h2a) 

        return E

    def estimate_beta(self, samples):
        assert(self.order == 1)
        h1, h2a, h2b, h3 = self.get_h_vals(samples)

        A = (2*h3 - h2b*(h2a+h2b))
        B = (2*h3 - 2 * h1 * h2a)
        C = (h2a+h2b - 2 * h1)

        self.beta = -np.roots([A.real, B.real, C.real]).min()

        E = h1 - self.beta *(h2a + h2b) + self.beta**2 * h3
        E /= (1 - self.beta * h1 + self.beta**2 * h2a) 

        return E

    @staticmethod
    @jit(nopython=True)
    def contract_inner_sum(vals, sections, arg):
        bottom_lim = 0
        for i in range(arg.size):
            arg[i] *= np.sum(vals[bottom_lim:sections[i]])
            bottom_lim = sections[i]
        return arg

    @property
    def state_dict(self):
        return self.machine.state_dict