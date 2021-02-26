import numpy as np
import netket as nk
from shutil import move
import sys
import mpi4py.MPI as mpi
import symmetries

N = int(sys.argv[1])
opt_par = int(sys.argv[2])
J2 = float(sys.argv[3])

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
        mats.append(((J1/4) * exchange))
        sites.append([i * L + j, i * L + (j+1)%L])
        mats.append(((J1/4) * exchange))
        sites.append([i * L + j, ((i+1)%L) * L + j])

if J2 != 0.0:
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

transl = symmetries.get_symms_square_lattice(L)

ma = nk.machine.QGPSSumSym(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
ma.init_random_parameters(sigma=0.05, start_from_uniform=False)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.01)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=2, n_chains=1)
sa.reset(True)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma, lsq_solver="cg")

arr = np.zeros(ma._epsilon.size, dtype=bool)
arr[:opt_par] = True

class PartialOpt(nk.Vmc):
    def iter(self, n_steps, step=1):
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                np.random.shuffle(arr)
                ma.change_opt_ids(arr.reshape(ma._epsilon.shape))
                dp = self._forward_and_backward()
                self.update_parameters(dp)
                if i == 0:
                    yield self.step_count


samples = 15000

# Create the optimization driver
gs = PartialOpt(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=sr, n_discard=50)

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


