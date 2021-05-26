import numpy as np
import netket as nk
import sys
from shutil import move
import mpi4py.MPI as mpi
import symmetries

import glob

from pathlib import Path
home = str(Path.home())

mode = 3
msr = True

rank = mpi.COMM_WORLD.Get_rank()


for i, N in enumerate([1,3]):
    en_loc = []
    en_loc_imag = []
    err_loc = []
    N_loc = []
    L_loc = []
    if rank == 0:
        with open("result_QGPS_N_{}.txt".format(N), "w") as fl:
            fl.write("L  E(real)  E(imag)  Error E\n")
    for L in [10, 30, 50, 70, 90, 110, 130, 150]:
        for fl in glob.glob(str(Path.home())+"/data/vGPS/heisenberg1D/Heisenberg_vGPS_netket_1D_{}_sites_N_{}_lin_model_stabilised/*/best_epsilon.npy".format(L, N)):
            print(fl)
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
            elif mode == 3:
                ma = nk.machine.QGPSLinExp(hi, n_bond_exp=0, n_bond_lin=N, automorphisms=transl, spin_flip_sym=True, dtype=complex)

            ma.init_random_parameters(sigma=0.02, start_from_uniform=False)

            eps = np.load(fl)
            ma._epsilon = eps
            ma._opt_params = ma._epsilon[ma._der_ids >= 0].copy()
            ma.reset()

            # Sampler
            sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=L,n_chains=1)
            sa.reset(True)

            est = nk.variational.estimate_expectations(ha, sa, 1000000//mpi.COMM_WORLD.size, n_discard=200)

            if rank == 0:
                with open("result_QGPS_N_{}.txt".format(N), "a") as fl:
                    fl.write("{}  {}  {}  {}\n".format(L, est.mean.real, est.mean.imag, est.error_of_mean))

