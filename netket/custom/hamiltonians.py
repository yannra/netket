import netket as nk
from netket.operator import LocalOperator
import numpy as np

class J1J2(LocalOperator):
    def __init__(self, g, J1=1.0, J2=0.0, msr=True):
        assert(len(g.length) == 2 or len(g.length) == 1)
        L = g.length[0]

        two_d = False
        if len(g.length) == 2:
            two_d = True

        # Sigma^z*Sigma^z interactions
        sigmaz = np.array([[1, 0], [0, -1]])
        mszsz = np.kron(sigmaz, sigmaz)

        # Exchange interactions
        exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

        # Couplings J1 and J2
        if two_d:
            mats = []
            sites = []
            for i in range(L):
                for j in range(L):
                    mats.append(((J1/4) * mszsz))
                    sites.append([i * L + j, i * L + (j+1)%L])
                    mats.append(((J1/4) * mszsz))
                    sites.append([i * L + j, ((i+1)%L) * L + j])
                    if msr:
                        mats.append((-(J1/4) * exchange))
                    else:
                        mats.append(((J1/4) * exchange))
                    sites.append([i * L + j, i * L + (j+1)%L])
                    if msr:
                        mats.append((-(J1/4) * exchange))
                    else:
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
        else:
            mats = []
            sites = []
            for i in range(L):
                mats.append(((J1/4) * mszsz))
                sites.append([i, (i+1)%L])
                if msr:
                    mats.append((-(J1/4) * exchange))
                else:
                    mats.append(((J1/4) * exchange))
                sites.append([i, (i+1)%L])

            if J2 != 0.0:
                for i in range(L):
                    mats.append(((J2/4) * mszsz))
                    sites.append([i, (i+2)%L])
                    mats.append(((J2/4) * exchange))
                    sites.append([i, (i+2)%L])

        # Spin based Hilbert Space
        hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

        # Custom Hamiltonian operator
        super().__init__(hi)
        for mat, site in zip(mats, sites):
            self += nk.operator.LocalOperator(hi, mat, site)