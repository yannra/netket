import numpy as np
import netket as nk
import sys
import scipy.optimize as spo
import netket.custom.utils as utls

from netket.utils import (
    MPI_comm as _MPI_comm,
    n_nodes as _n_nodes,
    node_number as _rank
)

from netket.operator import local_values as _local_values

# 1D Lattice
L = 40


g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

J1=1.
J2 = 0.0

ha = nk.custom.J1J2(g, J2=J2, msr=False)

hi = ha.hilbert

transl = nk.custom.get_symms_chain(L, point_group=True)
ma = nk.machine.QGPSProdSym(hi, n_bond=5, automorphisms=transl, spin_flip_sym=True, cluster_ids=None, dtype=complex)

ma._exp_kern_representation = False
ma._fast_update = False

t_state = nk.custom.TimeEvolvedState(ma, ha, beta=0.1, order=1)
t_state2 = nk.custom.TimeEvolvedState(t_state, ha, beta=0.1, order=1)

sa0 = nk.sampler.MetropolisExchange(ma,graph=g,d_max=L,n_chains=1)
sa_evolved = nk.sampler.MetropolisExchange(t_state2,graph=g,d_max=L,n_chains=1)
final_sampler = nk.custom.RandomSampler(t_state2, n_chains=1)

ma.init_random_parameters(sigma=0.1, start_from_uniform=False, small_arg=True)

alpha_guess = 1.0
noise_guess = 1.e-1

alph = np.ones((L, 2*ma._epsilon.shape[1]))*alpha_guess
noi = np.ones(L)*noise_guess

exp_lsq = []
exp_tst_lsq = []

en_fit = []

learning = nk.custom.QGPSLearning(ma)


for k in range(4000):
    en_fit.append(nk.variational.estimate_expectations(ha, sa0, 50).mean.real)
    if _rank == 0:
        print("Current En:", en_fit[-1])
    en = nk.variational.estimate_expectations(ha, sa_evolved, 50).mean.real
    if _rank == 0:
        print("Target En:", en)
    trn_basis = final_sampler.generate_samples(200).reshape(200, L)
    trn_amplitudes = np.exp(t_state2.log_val(trn_basis))
    trn_weightings = np.ones(trn_amplitudes.size)
    scale_factor = np.exp(learning.get_bias(trn_amplitudes, weightings=trn_weightings))
    trn_amplitudes /= scale_factor
    j = 0
    init_error = learning.mean_squared_error(trn_basis, trn_amplitudes, trn_weightings)/np.sum(trn_weightings)
    if _rank == 0:
        print("Init error", init_error, noi[0], np.sum(alph), np.sum(alph.flatten()>1.e6), en_fit[-1].real, flush=True)
    while((learning.mean_squared_error(trn_basis, trn_amplitudes, trn_weightings)/np.sum(trn_weightings) > 0.1 or j < 1) and j < 25):
        for i in range(L):
            learning.fit_step(trn_basis, trn_amplitudes, i, noise_tilde=noi, alpha=alph, opt_alpha=True, opt_noise=False, rvm=True, max_iterations=5)
            N = ma._epsilon.shape[1]
            for n in range(N):
                if np.sum(np.abs(ma._epsilon[i, n, :])) < 1.e-8:
                    if _rank == 0:
                        print("triggered")
                    ma._epsilon[i, n, :] = np.random.normal(loc=1.0, scale=0.01, size=2) + 1.j * np.random.normal(scale=0.01, size=2)
                    alph[i, 2 * n] = alpha_guess
                    alph[i, 2 * n + 1] = alpha_guess
            _MPI_comm.Bcast(ma._epsilon, root=0)
            ma._opt_params = ma._epsilon[ma._der_ids >= 0].copy()
            noi.fill(noi[i])
        trn_error = learning.mean_squared_error(trn_basis, trn_amplitudes, trn_weightings)/np.sum(trn_weightings)
        if _rank == 0:
            print("A", trn_error, noi[0], np.sum(alph), np.sum(alph.flatten()>1.e6), en_fit[-1].real, flush=True)
        alph.fill(alpha_guess)
        j += 1
    for i in range(L):
        learning.fit_step(trn_basis, trn_amplitudes, i, noise_tilde=noi, alpha=alph, opt_alpha=True, opt_noise=True, rvm=True, max_iterations=10)
        N = ma._epsilon.shape[1]
        for n in range(N):
            if np.sum(np.abs(ma._epsilon[i, n, :])) < 1.e-8:
                if _rank == 0:
                    print("triggered")
                ma._epsilon[i, n, :] = np.random.normal(loc=1.0, scale=0.01, size=2) + 1.j * np.random.normal(scale=0.01, size=2)
                alph[i, 2 * n] = alpha_guess
                alph[i, 2 * n + 1] = alpha_guess
        ma._opt_params = ma._epsilon[ma._der_ids >= 0].copy()
        noi.fill(noi[i])
    trn_error = learning.mean_squared_error(trn_basis, trn_amplitudes, trn_weightings)/np.sum(trn_weightings)
    if _rank == 0:
        print("B", trn_error, noi[0], np.sum(alph), np.sum(alph.flatten()>1.e6), en_fit[-1].real, flush=True)
    alph.fill(alpha_guess)
    j += 1
    print(np.sum(ma._epsilon), np.sum(learning.K))
