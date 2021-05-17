import numpy as np
import netket as nk
import sys
from shutil import move
import mpi4py.MPI as mpi
import symmetries

L = int(sys.argv[1])
N = int(sys.argv[2])
mode = int(sys.argv[3])
msr = bool(int(sys.argv[4]))

rank = mpi.COMM_WORLD.Get_rank()

if rank == 0:
    with open("result.txt", "w") as fl:
        fl.write("L, N, energy (real), energy (imag), energy_error\n")

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

ha = nk.custom.J1J2(g, J2=0.0, msr=msr)

transl = nk.custom.get_symms_chain(L)

if mode == 0:
    ma = nk.machine.QGPSSumSym(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
elif mode == 1:
    ma = nk.machine.QGPSProdSym(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
elif mode == 2:
    ma = nk.machine.QGPSProdSym(hi, n_bond=N, automorphisms=None, spin_flip_sym=False, dtype=complex)

ma.init_random_parameters(sigma=0.02, start_from_uniform=False)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.02)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=L,n_chains=1)
sa.reset(True)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma)

samples = 5000

# Create the optimization driver
gs = nk.custom.SweepOpt(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=sr, n_discard=20, max_opt=4000, check_improvement=False, reset_bias=False)

best_epsilon = ma._epsilon.copy()
best_en_upper_bound = None

if mpi.COMM_WORLD.Get_rank() == 0:
    with open("out.txt", "w") as fl:
        fl.write("")
    np.save("epsilon.npy", ma._epsilon)
    np.save("best_epsilon.npy", best_epsilon)

count = 0
for it in gs.iter(2000,1):
    if mpi.COMM_WORLD.Get_rank() == 0:
        move("epsilon.npy", "epsilon_old.npy")
        np.save("epsilon.npy", ma._epsilon)
        print(it,gs.energy, flush=True)
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(gs.energy.mean), np.imag(gs.energy.mean), gs.energy.error_of_mean))
        if best_en_upper_bound is None:
            best_en_upper_bound = gs.energy.mean.real + gs.energy.error_of_mean
        else:
            if (gs.energy.mean.real  + gs.energy.error_of_mean) < best_en_upper_bound:
                best_epsilon = ma._epsilon.copy()
                best_en_upper_bound = gs.energy.mean.real + gs.energy.error_of_mean
                np.save("best_epsilon.npy", best_epsilon)
        if it == 1959:
            best_en_upper_bound = None
    count += 1
    if count == 40:
        count = 0
        samples += 100
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
