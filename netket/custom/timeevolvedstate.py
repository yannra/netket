import netket.machine as machine
import numpy as np
from numba import jit


class TimeEvolvedState(machine.AbstractMachine):
    def __init__(self, QGPSAnsatz, hamiltonian, beta=1., order=1):
        super().__init__(QGPSAnsatz.hilbert, dtype=QGPSAnsatz.dtype)
        self.QGPS = QGPSAnsatz
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
        log_val_primes = [self.QGPS.log_val(x_primes)]
        mels = [np.ones(x.shape[0], dtype=np.float)]

        for j in range(self.order):
            sections.append(np.empty(x_primes.shape[0], dtype=np.int32))
            x_primes, mels_new = self.ha.get_conn_flattened(x_primes, sections[-1])
            mels.append(mels_new)
            log_val_primes.append(self.QGPS.log_val(x_primes))
 
        for j in range(self.order+1):
            if j != 0:
                arg = np.ones(log_val_primes[j-1].size, dtype=np.complex128)
                arg = self.contract_inner_sum((np.exp(log_val_primes[j])*mels[j]), sections[j], arg)

            else:
                arg = np.exp(log_val_primes[j])

            for k in range(j-1, 0, -1):
                arg_new = np.ones(log_val_primes[k-1].size, dtype=np.complex128)
                arg_new = self.contract_inner_sum((arg * mels[k]), sections[k], arg_new)
                bottom_lim = 0
                for l in range(arg_new.size):
                    arg_new[l] *= np.sum((arg * mels[k])[bottom_lim:sections[k][l]])
                    bottom_lim = sections[k][l]
                arg = arg_new

            out += ((-self.beta)**j/np.math.factorial(j)) * arg
        out = np.log(out)
        return out

    def der_log(self, x, out=None):
        return out

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
        return self.QGPS.state_dict