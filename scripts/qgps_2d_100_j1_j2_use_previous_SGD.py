import numpy as np
import netket as nk
from shutil import move
import sys
import mpi4py.MPI as mpi
import symmetries

N = int(sys.argv[1])
J2 = float(sys.argv[2])
cluster_edge = int(sys.argv[3])
epsilon_read_in = np.load(sys.argv[4])

J1 = 1.0

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

cluster_ids=[]
for i in range(cluster_edge):
    for j in range(cluster_edge):
        cluster_ids.append(i * L + j)

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

# Custom Hamiltonian operator
ha = nk.operator.LocalOperator(hi)
for mat, site in zip(mats, sites):
    ha += nk.operator.LocalOperator(hi, mat, site)

transl = symmetries.get_symms_square_lattice(L)

ma = nk.machine.QGPSSumSymExp(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, cluster_ids=cluster_ids, dtype=complex)
ma.init_random_parameters(sigma=1)

old_cluster_edge = int(np.sqrt(epsilon_read_in.shape[0]))

old_n_bond = min(epsilon_read_in.shape[1], ma._epsilon.shape[1])

ma._epsilon[:,:old_n_bond, :] = 0

count = 0
count_old = 0
for i in range(cluster_edge):
    for j in range(cluster_edge):
        if j < old_cluster_edge and i < old_cluster_edge:
            ma._epsilon[count,:old_n_bond, :] = epsilon_read_in[count_old, :old_n_bond, :]
            count_old += 1
        count += 1

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.005)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=2, n_chains=1)
sa.reset(True)

samples = 25000

# Create the optimization driver
gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=None, n_discard=100)

if mpi.COMM_WORLD.Get_rank() == 0:
    with open("out.txt", "w") as fl:
        fl.write("")

np.save("epsilon.npy", ma._epsilon)

for it in gs.iter(1950,1):
    if mpi.COMM_WORLD.Get_rank() == 0:
        move("epsilon.npy", "epsilon_old.npy")
        np.save("epsilon.npy", ma._epsilon)
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))

epsilon_avg = np.zeros(ma._epsilon.shape, dtype=ma._epsilon.dtype)

for it in gs.iter(50,1):
    epsilon_avg += ma._epsilon
    if mpi.COMM_WORLD.Get_rank() == 0:
        move("epsilon.npy", "epsilon_old.npy")
        np.save("epsilon.npy", ma._epsilon)
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))

epsilon_avg /= 50

ma._epsilon = epsilon_avg

est = nk.variational.estimate_expectations(ha, sa, 50000, n_discard=100)

if mpi.COMM_WORLD.Get_rank() == 0:
    np.save("epsilon_avg.npy", ma._epsilon)
    with open("result.txt", "a") as fl:
        fl.write("{}  {}  {}  {}\n".format(N, np.real(est.mean), np.imag(est.mean), est.error_of_mean))


