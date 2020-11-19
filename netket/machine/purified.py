# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jax.experimental.stax import Dense
from jax.experimental import stax
import jax
from collections import OrderedDict
from functools import reduce

from .abstract_machine import AbstractMachine

import numpy as _np
from netket.random import randint as _randint

from .jax import Jax
from netket.hilbert import JointHilbert


class PurifiedJaxMachine(Jax):
    def __init__(self, hilbert, *args, **kwargs):
        assert isinstance(hilbert, JointHilbert)

        super().__init__(hilbert, *args, **kwargs)

        def _ancilla_log_val(params, v, v_a):
            v_phys = v[:, 0 : hilbert.size_physical]

            x = jax.numpy.concatenate((v, v_a), axis=1)

            return self._forward_fn_nj(params, x).reshape(
                x.shape[0],
            )

        self._log_val_ancilla = jax.jit(_ancilla_log_val)

    def density_matrix(self, normalize=True):
        if self.hilbert.is_indexable:
            psi = self.log_val(self.hilbert.all_states())

            ancilla_hilb_size = self.hilbert.ancilla.n_states

            psi = psi.reshape((-1, ancilla_hilb_size))

            logmax = psi.real.max()
            psi = _np.exp(psi - logmax)

            rho = _np.matmul(psi, psi.conj().T)

            if normalize:
                norm = _np.trace(rho)
                rho /= norm

            return rho
        else:
            raise RuntimeError("The hilbert space is not indexable")

    def log_val_ancilla(self, x, a):
        return self._log_val_ancilla(self._params, x, a)

    def log_val(self, x, out=None):
        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        _out = self._forward_fn(self._params, x).reshape(
            x.shape[0],
        )

        if out is None:
            out = _out
        else:
            out[:] = _out

        return out

    @property
    def jax_forward(self):
        return self._forward_fn
