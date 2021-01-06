import numpy as np
import netket as nk
import sys
import mpi4py as mpi

N = int(sys.argv[1])
J2 = float(sys.argv[2])

J1 = 1.0

if mpi.MPI.COMM_WORLD.Get_rank() == 1:
    with open("result.txt", "w") as fl:
        fl.write("N, energy (real), energy (imag), energy_error\n")

L = 4
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

# Sigma^z*Sigma^z interactions
sigmaz = np.array([[1, 0], [0, -1]])
mszsz = np.kron(sigmaz, sigmaz)

# Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

# Couplings J1 and J2
mats = []
sites = []
for i in range(L):
    for j in range(L):
        mats.append(((J1/4) * mszsz))
        sites.append([i * L + j, i * L + (j+1)%L])
        mats.append(((J1/4) * mszsz))
        sites.append([i * L + j, ((i+1)%L) * L + j])
        mats.append((-(J1/4) * exchange))
        sites.append([i * L + j, i * L + (j+1)%L])
        mats.append((-(J1/4) * exchange))
        sites.append([i * L + j, ((i+1)%L) * L + j])

for i in range(L):
    for j in range(L):
        mats.append(((J2/4) * mszsz))
        sites.append([i * L + j, ((i+1)%L) * L + (j+1)%L])
        mats.append(((J2/4) * mszsz))
        sites.append([i * L + j, ((i+1)%L) * L + (j-1)%L])
        mats.append(((J2/4) * exchange))
        sites.append([i * L + j, ((i+1)%L) * L + (j+1)%L])
        mats.append(((J2/4) * exchange))
        sites.append([i * L + j, ((i+1)%L) * L + (j-1)%L])


# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

# Custom Hamiltonian operator
ha = nk.operator.LocalOperator(hi)
for mat, site in zip(mats, sites):
    ha += nk.operator.LocalOperator(hi, mat, site)



nk.exact.lanczos_ed(ha, compute_eigenvectors=False)


transl = []
for i in range(L**2):
    line = []
    col_id = i%L
    row_id = i//L
    for k in range(L):
        for l in range(L):
            line.append(L*row_id + col_id)
            col_id += 1
            if col_id == L:
                col_id = 0
        row_id += 1
        if row_id == L:
            row_id = 0
    transl.append(line)


epsilon = np.zeros((hi.size, N, 2), dtype=float)

for x in range(N):
    for j in range(0, hi.size):
        for k in range(2):
            epsilon[j, x, k] = 0.95 + 0.1*np.random.rand()

ma = nk.machine.QGPSPhaseSplitSumSym(hi, epsilon=epsilon, automorphisms=transl, spin_flip_sym=True)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.04)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma)

samples = max(4000, epsilon.size * 5)

# Create the optimization driver
gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=sr)

if mpi.MPI.COMM_WORLD.Get_rank() == 1:
    with open("out.txt", "w") as fl:
        fl.write("")

for it in gs.iter(1000,1):
    if mpi.MPI.COMM_WORLD.Get_rank() == 1:
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))

if mpi.MPI.COMM_WORLD.Get_rank() == 1:
    with open("result.txt", "a") as fl:
        fl.write("{}  {}  {}  {}\n".format(N, np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))


