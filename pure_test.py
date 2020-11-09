import netket as nk
import jax
from jax.experimental import stax
import numpy as np
from netket.machine import jax as nkj

L = 4
g = nk.graph.Grid([L])
hi = nk.hilbert.Spin(g, 0.5)

hi_a = nk.hilbert.PurifiedHilbert(hi)

# The hamiltonian
ha = nk.operator.LocalOperator(hi)
j_ops = []
for i in range(L):
    ha += (1.0 / 2.0) * nk.operator.spin.sigmax(hi, i)
    ha += (
        (2.0 / 4.0)
        * nk.operator.spin.sigmaz(hi, i)
        * nk.operator.spin.sigmaz(hi, (i + 1) % L)
    )
    j_ops.append(nk.operator.spin.sigmam(hi, i))

lind = nk.operator.LocalLiouvillian(ha, j_ops)

mod = stax.serial(stax.Dense(1 * hi.size), nkj.LogCoshLayer, nkj.SumLayer)

ma = nk.machine.PurifiedJaxMachine(hi_a, mod, dtype=complex)

sa = nk.sampler.MetropolisLocal(machine=ma)

v = sa.generate_samples(100)
vr = v.reshape(-1, hi_a.size)

eloc = nk.operator.local_values(ha, ma, vr)

v2 = sa.generate_samples(100)
v2r = v2.reshape(-1, hi_a.size)

vv = (vr, v2r)

lloc = nk.operator.local_values(lind, ma, vv)

op = nk.optimizer.Sgd(ma, learning_rate=0.001)

gs = nk.SteadyStatePure(lind, sa, op, n_samples=1000, sr=None, n_discard=100)

gs.run(1)
