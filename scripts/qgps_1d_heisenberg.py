import numpy as np
import netket as nk
import sys
import mpi4py.MPI as mpi
import symmetries

L = int(sys.argv[1])
N = int(sys.argv[2])
mode = int(sys.argv[3])

rank = mpi.COMM_WORLD.Get_rank()

if rank == 0:
    with open("result.txt", "w") as fl:
        fl.write("L, N, energy (real), energy (imag), energy_error\n")

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

ha = nk.operator.Heisenberg(hi, g, J=0.25)

transl = symmetries.get_symms_chain(L)

if mode == 0:
    ma = nk.machine.QGPSSumSym(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
elif mode == 1:
    ma = nk.machine.QGPSProdSym(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
elif mode == 2:
    ma = nk.machine.QGPSSumSymExp(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)
else:
    ma = nk.machine.QGPSProdSymExp(hi, n_bond=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)

ma.init_random_parameters(sigma=0.1)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.02)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=1,n_chains=1)
sa.reset(True)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma)

samples = max(5000, ma._epsilon.size * 5)

max_opt = 2500

arr = np.zeros(ma._epsilon.size, dtype=bool)
arr[:max_opt] = True
max_id = min(max_opt, arr.size)

class SiteSweepOpt(nk.Vmc):
    def iter(self, n_steps, step=1):
        global max_id
        count = 0
        for _ in range(0, n_steps, step):
            for i in range(0, step):
                ma.change_opt_ids(arr.reshape(ma._epsilon.shape))
                dp = self._forward_and_backward()
                self.update_parameters(dp)
                arr.fill(False)
                arr[max_id:(max_id+max_opt)] = True
                if max_id + max_opt > arr.size:
                    arr[:(max_id + max_opt - arr.size)] = True
                    max_id = min((max_id + max_opt - arr.size), arr.size)
                else:
                    max_id = min((max_id + max_opt), arr.size)
                if i == 0:
                    yield self.step_count
                count += 1


# Create the optimization driver
gs = SiteSweepOpt(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=samples, sr=sr)

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
