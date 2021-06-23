import netket.sampler as samp
import numpy as np

class RandomSampler(samp.AbstractSampler):
    def __init__(self, machine, n_chains):
        super().__init__(machine, n_chains)
        self._state = np.zeros((n_chains, self._input_size))
        self.reset(True)
    def reset(self, init_random=False):
        for i in range(self.sample_size):
            self._machine.hilbert.random_state(out=self._state[i])
    def __next__(self):
        for i in range(self.sample_size):
            self._machine.hilbert.random_state(out=self._state[i])
        return self._state