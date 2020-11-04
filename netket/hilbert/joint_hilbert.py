from .abstract_hilbert import AbstractHilbert

import numpy as _np
from netket import random as _random
import netket as nk


class JointHilbert(AbstractHilbert):
    r"""Superoperatorial hilbert space for states living in the
    tensorised state H\otimes H, encoded according to Choi's isomorphism."""

    def __init__(self, hilb_1, hilb_2):
        r"""Superoperatorial hilbert space for states living in the
           tensorised state H\otimes H, encoded according to Choi's isomorphism.

        Args:
            hilb: the hilbrt space H.

        Examples:
            Simple superoperatorial hilbert space for few spins.

           >>> from netket.graph import Hypercube
           >>> from netket.hilbert import Spin, DoubledHilbert
           >>> g = Hypercube(length=5,n_dim=2,pbc=True)
           >>> hi = Spin(graph=g, s=0.5)
           >>> hi2 = DoubledHilbert(hi)
           >>> print(hi2.size)
           50
        """
        if hasattr(hilb_1, "graph") and hasattr(hilb_2, "graph"):
            joint_graph = nk.graph.disjoint_union(hilb_1.graph, hilb_2.graph)
        else:
            raise ErrorException("No graph")

        if not hilb_1.local_states == hilb_2.local_states:
            raise ValueError("Hilbert spaces do not have the same local basis")

        self.graph = joint_graph
        self.physical = hilb_1
        self.ancilla = hilb_2

        self._size = hilb_1.size + hilb_2.size

        super().__init__()

    @property
    def size(self):
        return self._size

    @property
    def is_discrete(self):
        return self.physical.is_discrete and self.ancilla.is_discrete

    @property
    def is_finite(self):
        return self.physical.is_finite and self.ancilla.is_finite

    @property
    def local_size(self):
        return self.physical.local_size

    @property
    def local_states(self):
        return self.physical.local_states

    @property
    def size_physical(self):
        return self.physical.size

    @property
    def size_ancilla(self):
        return self.ancilla.size

    @property
    def n_states(self):
        return self.physical.n_states * self.ancilla.n_states

    def numbers_to_states(self, numbers, out=None):
        if out is None:
            out = _np.empty((numbers.shape[0], self._size))

        # !!! WARNING
        # This code assumes that states are stored in a MSB
        # (Most Significant Bit) format.
        # We assume that the rightmost-half indexes the LSBs
        # and the leftmost-half indexes the MSBs
        # HilbertIndex-generated states respect this, as they are:
        # 0 -> [0,0,0,0]
        # 1 -> [0,0,0,1]
        # 2 -> [0,0,1,0]
        # etc...

        n = self.physical.size
        na = self.ancilla.size
        dim_physical = self.physical.n_states
        dim_ancilla = self.ancilla.n_states
        i_phys, i_anc = _np.divmod(numbers, dim_ancilla)

        self.physical.numbers_to_states(i_phys, out=out[:, 0:n])
        self.ancilla.numbers_to_states(i_anc, out=out[:, n : n + na])

        return out

    def states_to_numbers(self, states, out=None):
        if out is None:
            out = _np.empty(states.shape[0], _np.int64)

        # !!! WARNING
        # See note above in numbers_to_states

        n = self.physical.size
        na = self.ancilla.size
        dim_physical = self.physical.n_states
        dim_ancilla = self.ancilla.n_states

        self.physical.states_to_numbers(states[:, 0:n], out=out)
        _out_l = out * dim_ancilla
        self.ancilla.states_to_numbers(states[:, n : n + na], out=out)
        out += _out_l

        return out

    def random_state(self, out=None, rgen=None):
        if out is None:
            out = _np.empty(self._size)

        n = self.physical.size
        na = self.ancilla.size

        self.physical.random_state(out=out[0:n], rgen=rgen)
        self.ancilla.random_state(out=out[n : n + na], rgen=rgen)

        return out

    def __repr__(self):
        return "JointHilbert({}, {})".format(self.physical, self.ancilla)


def PurifiedHilbert(hilb, Na=None):
    if Na is None:
        Na = hilb.size

    a_graph = nk.graph.Edgeless(Na)

    if isinstance(hilb, nk.hilbert.Spin):
        a_hilb = nk.hilbert.Spin(a_graph, hilb._s)
    elif isinstance(hilb, nk.hilbert.CustomHilbert):
        a_hilb = nk.hilbert.CustomHilbert(a_graph, local_states=hilb.local_states)
    else:
        raise ErrorException("Unrecognized hilb type {}".format(hilb))

    return JointHilbert(hilb, a_hilb)
