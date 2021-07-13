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
L = 20


if _rank == 0:
    with open("out.txt", "w") as fl:
        fl.write("")


g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

J1=1.
J2 = 0.0

ha = nk.custom.J1J2(g, J2=J2, msr=False)

hi = ha.hilbert

transl = nk.custom.get_symms_chain(L, point_group=True)
ma = nk.machine.QGPSLinExp(hi, n_bond_lin=1, n_bond_exp=1, automorphisms=transl, spin_flip_sym=True, cluster_ids=None, dtype=complex)
ma._exp_kern_representation = False
ma._fast_update = False
ma.init_random_parameters(sigma=3.0, start_from_uniform=False, small_arg=False)

ma._epsilon[0, ma._n_bond_lin:,:] = 0.0
ma._opt_params = ma._epsilon[ma._der_ids >= 0].copy()


t_state = nk.custom.TimeEvolvedState(ma, ha, beta=0.1, order=1)
t_state2 = nk.custom.TimeEvolvedState(t_state, ha, beta=0.1, order=1)

sa0 = nk.sampler.MetropolisExchange(ma,graph=g,d_max=L,n_chains=1)
sa_evolved = nk.sampler.MetropolisExchange(t_state2,graph=g,d_max=L,n_chains=1)
final_sampler = nk.custom.RandomSampler(t_state2, n_chains=1)


exp_lsq = []
exp_tst_lsq = []

en_fit = []

learninglin = nk.custom.QGPSLearningLin(ma)
learningexp = nk.custom.QGPSLearningExp(ma)

learninglin.noise = 0.00001

learningexp.noise_tilde = 0.001

alpha_guess = 1.0

for k in range(4000):
    # ma._fast_update = True
    ma.reset()
    current_en = nk.variational.estimate_expectations(ha, sa0, 50)
    en_fit.append(current_en.mean.real)
    if _rank == 0:
        print("Current En:", en_fit[-1])
        with open("out.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(np.real(current_en.mean), np.imag(current_en.mean), current_en.error_of_mean))
    en = nk.variational.estimate_expectations(ha, sa_evolved, 50).mean.real
    if _rank == 0:
        print("Target En:", en)
    trn_basis = final_sampler.generate_samples(2000).reshape(2000, L)
    test_basis = final_sampler.generate_samples(50).reshape(50, L)
    trn_amplitudes = np.exp(t_state2.log_val(trn_basis))
    test_amplitudes = np.exp(t_state2.log_val(test_basis))
    # ma._fast_update = False
    trn_weightings = np.ones(trn_amplitudes.size)
    test_weightings = np.ones(test_amplitudes.size)
    scale_factor = np.exp(learningexp.get_bias(trn_amplitudes, weightings=trn_weightings))
    trn_amplitudes /= scale_factor
    test_amplitudes /= scale_factor
    j = 0
    init_error_trn = learninglin.mean_squared_error(trn_basis, trn_amplitudes, trn_weightings)/(_n_nodes * len(trn_amplitudes))
    init_error_test = learninglin.mean_squared_error(test_basis, test_amplitudes, test_weightings)/(_n_nodes * len(test_amplitudes))
    if _rank == 0:
        print("Init error", init_error_trn, init_error_test, flush=True)
    log_ml = 1.0
    log_ml_old = 1.0
    # while ((log_ml - log_ml_old)/np.sqrt(abs(log_ml*log_ml_old)) > 1.e-3 or j < 2):
    while j < 10:
        log_ml_old = log_ml
        log_ml = 0.0
        for i in range(L):
            learninglin.fit_step(trn_basis, trn_amplitudes, i, opt_alpha=True, opt_noise=True, rvm=True, multiplication=learningexp.predict(trn_basis))
            log_ml += learninglin.log_marg_lik()
            learninglin.alpha_mat[i, ~learninglin.active_elements] = 1000 * alpha_guess
            learningexp.fit_step(trn_basis, trn_amplitudes, i, opt_alpha=True, opt_noise=True, rvm=True, multiplication=learninglin.predict(trn_basis))
            log_ml += learningexp.log_marg_lik()
            learningexp.alpha_mat[i, ~learningexp.active_elements] = 1000 * alpha_guess
            N = ma._epsilon.shape[1]
            for n in range(ma._epsilon.shape[1]):
                if np.sum(np.abs(ma._epsilon[i, n, :])) < 1.e-8:
                    if _rank == 0:
                        if n < ma._n_bond_lin:
                            print("triggered lin")
                        else:
                            print("triggered exp")
                    ma._epsilon[i, n, :] = np.random.normal(loc=1.0, scale=3.0, size=2) + 1.j * np.random.normal(scale=0.01, size=2)
                    if n < ma._n_bond_lin:
                        learninglin.alpha_mat[i, 2 * n] = alpha_guess
                        learninglin.alpha_mat[i, 2 * n + 1] = alpha_guess
                    else:
                        n_id = n % ma._n_bond_lin
                        learningexp.alpha_mat[i, 2 * n_id] = alpha_guess
                        learningexp.alpha_mat[i, 2 * n_id + 1] = alpha_guess
            _MPI_comm.Bcast(ma._epsilon, root=0)
        trn_error = learninglin.mean_squared_error(trn_basis, trn_amplitudes, trn_weightings)/(_n_nodes * len(trn_amplitudes))
        test_error = learninglin.mean_squared_error(test_basis, test_amplitudes, test_weightings)/(_n_nodes * len(test_amplitudes))
        if _rank == 0:
            print(trn_error, test_error, (log_ml), (log_ml - log_ml_old)/np.sqrt(abs(log_ml*log_ml_old)), learninglin.noise, learningexp.noise_tilde, en_fit[-1].real, flush=True)
        j += 1
        ma._opt_params = ma._epsilon[ma._der_ids >= 0].copy()

