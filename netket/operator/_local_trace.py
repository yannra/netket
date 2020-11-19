import numpy as _np
from numba import jit
import netket

from ._local_liouvillian import LocalLiouvillian as _LocalLiouvillian
from netket.machine.density_matrix.abstract_density_matrix import (
    AbstractDensityMatrix as DensityMatrix,
)

if netket.utils.jax_available:
    from netket.machine import PurifiedJaxMachine
else:

    class PurifiedJaxMachine:
        pass


@jit(nopython=True)
def _local_trace_purified_kernel(logψ_σ_a, logψ_η_a, logψ_ν_a, mels, sections, out):
    low_range = 0
    for i, s in enumerate(sections):
        out[i] = (
            mels[low_range:s]
            * _np.exp(
                logψ_η_a[low_range:s]
                + _np.conj(logψ_ν_a[low_range:s])
                - 2 * (_np.real(logψ_σ_a[i]))
            )
        ).sum()
        low_range = s


# For expectation values of lindblad
def _local_traces_purified_impl(op, machine, σ_a, logψ_σ_a, out):

    hi = machine.hilbert
    Np = hi.size_physical
    Na = hi.size_ancilla

    # Extract the /sigma part of the state (the physical)
    σ = _np.asarray(σ_a[:, 0:Np])

    # Extract the ancilla part of the state
    a = σ_a[:, Np : Np + Na]

    sections = _np.empty(σ.shape[0], dtype=_np.int32)
    η, ν, mels = op._get_conn_flattened_(σ, σ, sections, pad=False, merge_output=False)

    # sections are indices like in a sparse matrix and they don't have a leading 0.
    # Generate the length of each section
    section_lengths = _np.diff(sections, prepend=0)

    # Repeat every ancilla a sufficient number of times
    a_ext = _np.repeat(a, section_lengths, axis=0)

    # join ancillas and states
    # η_a = _np.concatenate((η, a_ext), axis=1)
    # ν_a = _np.concatenate((ν, a_ext), axis=1)

    logψ_η_a = machine.log_val_ancilla(η, a_ext)
    logψ_ν_a = machine.log_val_ancilla(ν, a_ext)

    _local_trace_purified_kernel(
        _np.asarray(logψ_σ_a),
        _np.asarray(logψ_η_a),
        _np.asarray(logψ_ν_a),
        mels,
        sections,
        out,
    )


def local_trace(op, machine, v, log_vals=None, out=None):
    r"""
    Computes local values of the operator `op` for all `samples`.

    The local value is defined as
    .. math:: O_{\mathrm{loc}}(x) = \langle x | O | \Psi \rangle / \langle x | \Psi \rangle


            Args:
                op: Hermitian operator.
                v: A numpy array or matrix containing either a batch of visible
                    configurations :math:`V = v_1,\dots v_M`.
                    Each row of the matrix corresponds to a visible configuration.
                machine: Wavefunction :math:`\Psi`.
                log_vals: A scalar/numpy array containing the value(s) :math:`\Psi(V)`.
                    If not given, it is computed from scratch.
                    Defaults to None.
                out: A scalar or a numpy array of local values of the operator.
                    If not given, it is allocated from scratch and then returned.
                    Defaults to None.

            Returns:
                If samples is given in batches, a numpy array of local values
                of the operator, otherwise a scalar.
    """

    if log_vals is None:
        log_vals = machine.log_val(v)

    # True when this is the local_value of a densitymatrix times an operator (observable)
    _impl = _local_traces_purified_impl

    assert (
        v.shape[1] == op.hilbert.size
    ), "samples has wrong shape: {}; expected (?, {})".format(v.shape, op.hilbert.size)
    if v.ndim != 2:
        raise RuntimeError("Invalid input shape, expected a 2d array")

    if out is None:
        Nb = v.shape[0]

        out = _np.empty(Nb, dtype=_np.complex128)

    _impl(op, machine, v, log_vals, out)

    return out
