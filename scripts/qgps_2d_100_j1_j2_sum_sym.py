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

transl = nk.custom.get_symms_square_lattice(L)

ma = nk.machine.QGPSSumSym(ha.hilbert, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
ma.init_random_parameters(sigma=0.02, start_from_uniform=False)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.02)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=2, n_chains=1)
sa.reset(True)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma)


samples = 2000

# Create the optimization driver
gs = nk.custom.SweepOpt(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=sr, n_discard=100)

best_epsilon = ma._epsilon.copy()
best_en_upper_bound = None

if mpi.COMM_WORLD.Get_rank() == 0:
    with open("out.txt", "w") as fl:
        fl.write("")
    np.save("epsilon.npy", ma._epsilon)
    np.save("best_epsilon.npy", best_epsilon)

count = 0
for it in gs.iter(3000,1):
    if mpi.COMM_WORLD.Get_rank() == 0:
        move("epsilon.npy", "epsilon_old.npy")
        np.save("epsilon.npy", ma._epsilon)
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))
        if best_en_upper_bound is None:
            best_en_upper_bound = gs.energy.mean.real + gs.energy.error_of_mean
        else:
            if (gs.energy.mean.real  + gs.energy.error_of_mean) < best_en_upper_bound:
                best_epsilon = ma._epsilon.copy()
                best_en_upper_bound = gs.energy.mean.real + gs.energy.error_of_mean
                np.save("best_epsilon.npy", best_epsilon)
    count += 1
    if count == 10:
        count = 0
        gs.n_samples = gs.n_samples + 50

mpi.COMM_WORLD.Bcast(best_epsilon, root=0)

ma._epsilon = best_epsilon
est = nk.variational.estimate_expectations(ha, sa, 50000, n_discard=100)

if mpi.COMM_WORLD.Get_rank() == 0:
    with open("result.txt", "a") as fl:
        fl.write("{}  {}  {}  {}\n".format(N, np.real(est.mean), np.imag(est.mean), est.error_of_mean))


