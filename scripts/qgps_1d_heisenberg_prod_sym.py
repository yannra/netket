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


epsilon = np.zeros((hi.size, N, 2), dtype=float)

for x in range(N):
    for j in range(0, hi.size):
        for k in range(2):
            epsilon[j, x, k] = 0.95 + 0.1*np.random.rand()

ma = nk.machine.QGPSProdSym(hi, epsilon=epsilon, automorphisms=transl, spin_flip_sym=True, dtype=float)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.04)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma, diag_shift=0.005)

samples = max(4000, epsilon.size * 5)

# Create the optimization driver
gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=sr)

if rank == 1:
    with open("out.txt", "w") as fl:
        fl.write("")

for it in gs.iter(2000,1):
    if rank == 0:
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))

if rank == 0:
    with open("result.txt", "a") as fl:
        fl.write("{}  {}  {}  {}  {}\n".format(L, N, np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))


