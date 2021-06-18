import numpy as np
import netket as nk
import sys
import shutil
from shutil import move
import mpi4py.MPI as mpi
import symmetries
import os


N = 32
L = 10
mode = 0
J2 = 0.5

rank = mpi.COMM_WORLD.Get_rank()

initial_folder = "/home/mmm0475/Scratch/J1_J2_2D_10_by_10_J2_0.5_sum_sym_N_32_large_sample_with_bias_345550"

if rank == 0:
    for item in os.listdir(initial_folder):
        s = os.path.join(initial_folder, item)
        d = os.path.join("", "OLD_"+item)
        if not os.path.isdir(s):
            shutil.copy2(s, d)
    shutil.copyfile("OLD_epsilon.npy", "epsilon.npy")
    shutil.copyfile("OLD_epsilon_old.npy", "epsilon_old.npy")

mpi.COMM_WORLD.barrier()

opt_process = np.genfromtxt("OLD_out.txt")


g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

ha = nk.custom.J1J2(g, J2=J2, msr=True)

transl = nk.custom.get_symms_square_lattice(L)

if mode == 0:
    ma = nk.machine.QGPSSumSym(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
elif mode == 1:
    ma = nk.machine.QGPSProdSym(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
elif mode == 2:
    ma = nk.machine.QGPSProdSym(hi, n_bond=N, automorphisms=None, spin_flip_sym=False, dtype=complex)

ma._exp_kern_representation = False

if L > 8 and N > 10:
    ma.init_random_parameters(sigma=0.01, start_from_uniform=False)
else:
    ma.init_random_parameters(sigma=0.02, start_from_uniform=False)

ma.bias = -100.

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.02)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=2*L,n_chains=1)
sa.reset(True)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma)

samples = 10100

# Create the optimization driver
gs = nk.custom.SweepOpt(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=sr, n_discard=20, max_opt=6400, check_improvement=False, reset_bias=True)

eps = np.load("OLD_epsilon_old.npy")
ma._epsilon = eps.copy()
ma._opt_params = ma._epsilon[ma._der_ids >= 0].copy()
ma.reset()

best_epsilon = ma._epsilon.copy()

np.save("epsilon.npy", ma._epsilon)
np.save("epsilon_old.npy", ma._epsilon)

best_en_upper_bound = min(opt_process[:,0] + opt_process[:,2])

if rank == 0:
    with open("out.txt", "w") as fl:
        fl.write("")

est = nk.variational.estimate_expectations(ha, sa, 10000//mpi.COMM_WORLD.size, n_discard=200)

if rank == 0:
    print("Init calc:", est.mean, flush=True)

count = 0
total_count = 0
for i in range(opt_process.shape[0]-2):
    if mpi.COMM_WORLD.Get_rank() == 0:
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(opt_process[i,0], opt_process[i,1], opt_process[i,2]))
        if i == 2959:
            best_en_upper_bound = None
    count += 1
    total_count += 1
    if count == 40:
        count = 0
        samples += 100
        sr._diag_shift *= 0.97
        gs.n_samples = samples

for it in gs.iter(3000 - total_count,1):
    if mpi.COMM_WORLD.Get_rank() == 0:
        if it >= 1:
            move("epsilon.npy", "epsilon_old.npy")
            np.save("epsilon.npy", ma._epsilon)
        print(it+total_count,gs.energy, flush=True)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))
        if best_en_upper_bound is None:
            best_en_upper_bound = gs.energy.mean.real + gs.energy.error_of_mean
        else:
            if (gs.energy.mean.real  + gs.energy.error_of_mean) < best_en_upper_bound:
                best_epsilon = ma._epsilon.copy()
                best_en_upper_bound = gs.energy.mean.real + gs.energy.error_of_mean
                np.save("best_epsilon.npy", best_epsilon)
        if it+total_count == 2959:
            best_en_upper_bound = None
    count += 1
    if count == 40:
        count = 0
        samples += 100
        sr._diag_shift *= 0.97
        gs.n_samples = samples


mpi.COMM_WORLD.Bcast(best_epsilon, root=0)

mpi.COMM_WORLD.barrier()

ma._epsilon = best_epsilon
ma._opt_params = ma._epsilon[ma._der_ids >= 0].copy()
ma.reset()

est = nk.variational.estimate_expectations(ha, sa, 100000//mpi.COMM_WORLD.size, n_discard=200)

if mpi.COMM_WORLD.Get_rank() == 0:
    with open("result.txt", "a") as fl:
        fl.write("{}  {}  {}  {}  {}\n".format(L, N, np.real(est.mean), np.imag(est.mean), est.error_of_mean))
