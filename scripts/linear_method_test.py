from netket.stats import sum_inplace as _sum_inplace, mean as _mean
import numpy as np
import netket as nk
from netket.custom.hamiltonians import J1J2
import netket.custom.symmetries as symms
import mpi4py.MPI as mpi

import scipy




from netket.custom import local_values_with_der

from netket.operator import local_values

import sys

# 1D Lattice
L = 4


g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
# Exchange interactions
J1=1.
J2 = 0.5

ha = J1J2(g, J1=J1, J2=J2)
hi = ha.hilbert

# nk.exact.lanczos_ed(ha)

transl = symms.get_symms_square_lattice(L, point_group=True)
ma = nk.machine.QGPSSumSym(hi, n_bond=5, automorphisms=transl, spin_flip_sym=True, cluster_ids=None, dtype=complex)
ma._exp_kern_representation = True
ma.init_random_parameters(sigma=0.01, start_from_uniform=False)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=1)
# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma, graph=g, n_chains=1)

sa.reset(init_random=True)

# Stochastic Reconfiguration

# Create the optimization driver
gs = nk.custom.SweepOptLinMethod(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=4000, n_discard=50, epsilon=0.5)

gs._stab_shift = 0.1

for it in gs.iter(50,1):
    if mpi.COMM_WORLD.Get_rank() == 0:
        print(it,gs.energy)
        print(gs._stab_shift)