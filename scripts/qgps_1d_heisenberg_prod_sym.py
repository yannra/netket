import numpy as np
import netket as nk
import sys
import mpi4py as mpi

L = int(sys.argv[1])
N = int(sys.argv[2])

rank = mpi.MPI.COMM_WORLD.Get_rank()

if rank == 0:
    with open("result.txt", "w") as fl:
        fl.write("L, N, energy (real), energy (imag), energy_error\n")

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

ha = nk.operator.Heisenberg(hi, g, J=0.25)

transl = []
for i in range(L):
    line = []
    for k in range(L):
        line.append((i+k)%L)
    transl.append(line)

ma = nk.machine.QGPSProdSym(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True)

ma.init_random_parameters(sigma=0.1)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.02)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma)

samples = max(5000, ma._epsilon.size * 5)

# Create the optimization driver
gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=sr, n_discard=50)

if mpi.COMM_WORLD.Get_rank() == 0:
    with open("out.txt", "w") as fl:
        fl.write("")

for it in gs.iter(1950,1):
    if mpi.COMM_WORLD.Get_rank() == 0:
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))

epsilon_avg = np.zeros(ma._epsilon.shape, dtype=ma._epsilon.dtype)

for it in gs.iter(50,1):
    epsilon_avg += ma._epsilon
    if mpi.COMM_WORLD.Get_rank() == 0:
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))

epsilon_avg /= 50

ma._epsilon = epsilon_avg

sa = nk.sampler.MetropolisExchange(machine=ma,graph=g)
est = nk.variational.estimate_expectations(ha, sa, 50000, n_discard=100)

if mpi.COMM_WORLD.Get_rank() == 0:
    np.save("epsilon_avg.npy", ma._epsilon)
    with open("result.txt", "a") as fl:
        fl.write("{}  {}  {}  {}  {}\n".format(L, N, np.real(est.mean), np.imag(est.mean), est.error_of_mean))