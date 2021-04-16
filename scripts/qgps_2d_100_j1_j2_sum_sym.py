import numpy as np
import netket as nk
from shutil import move
import sys
import mpi4py.MPI as mpi
import symmetries

N = int(sys.argv[1])
J2 = float(sys.argv[2])

if mpi.COMM_WORLD.Get_rank() == 0:
    with open("result.txt", "w") as fl:
        fl.write("N, energy (real), energy (imag), energy_error\n")

L = 10
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

# Sigma^z*Sigma^z interactions
sigmaz = np.array([[1, 0], [0, -1]])
mszsz = np.kron(sigmaz, sigmaz)

# Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

ha = nk.custom.J1J2(g, J2=J2, msr=False)

transl = symmetries.get_symms_square_lattice(L)

ma = nk.machine.QGPSSumSym(ha.hilbert, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
ma._exp_kern_representation = False
ma.init_random_parameters(sigma=0.05, start_from_uniform=False)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.02)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=2, n_chains=1)
sa.reset(True)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma)


max_opt = 4000

arr = np.zeros(ma._epsilon.size, dtype=bool)
arr[:max_opt] = True
max_id = min(max_opt, arr.size)

samples = 10000

# Create the optimization driver
gs = nk.custom.SweepOptStabSR(hamiltonian=ha, sampler=sa, n_samples=samples, sr=sr, n_discard=50, search_radius=2)

if mpi.COMM_WORLD.Get_rank() == 0:
    with open("out.txt", "w") as fl:
        fl.write("")
    with open("sr_par.txt", "w") as fl:
        fl.write("")

np.save("epsilon.npy", ma._epsilon)

for it in gs.iter(3950,1):
    if mpi.COMM_WORLD.Get_rank() == 0:
        move("epsilon.npy", "epsilon_old.npy")
        np.save("epsilon.npy", ma._epsilon)
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))
        with open("sr_par.txt", "a") as fl:
            fl.write("{}  {}\n".format(gs._diag_shift, gs._time_step))
    if gs._diag_shift > 1:
        gs._diag_shift = 0.01
    if gs._time_step < 0.001:
        gs._time_step = 0.02

epsilon_avg = np.zeros(ma._epsilon.shape, dtype=ma._epsilon.dtype)

for it in gs.iter(50,1):
    epsilon_avg += ma._epsilon
    if mpi.COMM_WORLD.Get_rank() == 0:
        move("epsilon.npy", "epsilon_old.npy")
        np.save("epsilon.npy", ma._epsilon)
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))
        with open("sr_par.txt", "a") as fl:
            fl.write("{}  {}\n".format(gs._diag_shift, gs._time_step))
    if gs._diag_shift > 1:
        gs._diag_shift = 0.01
    if gs._time_step < 0.001:
        gs._time_step = 0.02

epsilon_avg /= 50

ma._epsilon = epsilon_avg

est = nk.variational.estimate_expectations(ha, sa, 50000, n_discard=100)

if mpi.COMM_WORLD.Get_rank() == 0:
    np.save("epsilon_avg.npy", ma._epsilon)
    with open("result.txt", "a") as fl:
        fl.write("{}  {}  {}  {}\n".format(N, np.real(est.mean), np.imag(est.mean), est.error_of_mean))


