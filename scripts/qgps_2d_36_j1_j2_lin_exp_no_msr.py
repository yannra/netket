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

L = 6
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

ha = nk.custom.J1J2(g, J2=J2, msr=False)

transl = symmetries.get_symms_square_lattice(L)

ma = nk.machine.QGPSLinExp(ha.hilbert, n_bond_lin=N, n_bond_exp=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)

ma.init_random_parameters(sigma=0.05, start_from_uniform=False)

ma._epsilon[0,:,:] -= 1
ma._opt_params = ma._epsilon[ma._der_ids >= 0].copy()

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.02)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=2, n_chains=1)
sa.reset(True)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma)

samples = 7500

gs = nk.custom.SweepOpt(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=sr, n_discard=50, sweep_by_bonds=True)


if mpi.COMM_WORLD.Get_rank() == 0:
    with open("out.txt", "w") as fl:
        fl.write("")

samp = max(2450, ma._epsilon.size//2)

for it in gs.iter(samp,1):
    if mpi.COMM_WORLD.Get_rank() == 0:
        np.save("epsilon.npy", ma._epsilon)
        print(it,gs.energy)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))

epsilon_avg = np.zeros(ma._epsilon.shape, dtype=ma._epsilon.dtype)

for it in gs.iter(50,1):
    epsilon_avg += ma._epsilon
    if mpi.COMM_WORLD.Get_rank() == 0:
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


