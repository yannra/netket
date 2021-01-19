import numpy as np
import netket as nk
import sys
import mpi4py.MPI as mpi
import symmetries

N = int(sys.argv[1])
J2 = float(sys.argv[2])

J1 = 1.0

if mpi.COMM_WORLD.Get_rank() == 0:
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


transl = symmetries.get_symms_square_lattice(L)

ma = nk.machine.QGPSPhaseSplitSumSym(hi, n_bond_amplitude=N//2, n_bond_phase=N//2, automorphisms=transl, spin_flip_sym=True)

ma.init_random_parameters(sigma=0.1)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.04)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=2)

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

sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=2)
est = nk.variational.estimate_expectations(ha, sa, 50000, n_discard=100)

if mpi.COMM_WORLD.Get_rank() == 0:
    np.save("epsilon_avg.npy", ma._epsilon)
    with open("result.txt", "a") as fl:
        fl.write("{}  {}  {}  {}\n".format(N, np.real(est.mean), np.imag(est.mean), est.error_of_mean))
