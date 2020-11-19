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
def _local_values_kernel(log_vals, log_val_primes, mels, sections, out):
    low_range = 0
    for i, s in enumerate(sections):
        out[i] = (
            mels[low_range:s] * _np.exp(log_val_primes[low_range:s] - log_vals[i])
        ).sum()
        low_range = s


def _local_values_impl(op, machine, v, log_vals, out):

    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_primes, mels = op.get_conn_flattened(_np.asarray(v), sections)

    log_val_primes = machine.log_val(v_primes)

    _local_values_kernel(
        _np.asarray(log_vals), _np.asarray(log_val_primes), mels, sections, out
    )


# For expectation values of observables
def _local_values_purified_op_impl(op, machine, σ_a, log_vals, out):

    hi = machine.hilbert
    # Extract the /sigma part of the state (the physical)
    σ = _np.asarray(σ_a[:, 0 : hi.size_physical])
    # Extract the ancilla part of the state
    a = σ_a[:, hi.size_physical : hi.size_physical + hi.size_ancilla]

    sections = _np.empty(σ.shape[0], dtype=_np.int32)
    σ_primes, mels = op.get_conn_flattened(σ, sections)

    # sections are indices like in a sparse matrix and they don't have a leading 0.
    # Generate the length of each section
    section_lengths = _np.diff(sections, prepend=0)

    # Repeat every ancilla a sufficient number of times
    a_primes = _np.repeat(a, section_lengths, axis=0)

    # join ancillas and states
    σ_primes_a = _np.concatenate((σ_primes, a_primes), axis=1)

    log_val_primes = machine.log_val(σ_primes_a)

    _local_values_kernel(
        _np.asarray(log_vals), _np.asarray(log_val_primes), mels, sections, out
    )


@jit(nopython=True)
def _local_values_purified_kernel(
    logψ_σ_a, logψ_η_b, logψ_σ_primes_b, logψ_η_primes_b, logψ_η_a, mels, sections, out
):
    low_range = 0
    for i, s in enumerate(sections):
        out[i] = (
            mels[low_range:s]
            * _np.exp(
                logψ_σ_primes_b[low_range:s]
                + _np.conj(logψ_η_primes_b[low_range:s])
                + logψ_η_a[i]
                - (logψ_σ_a[i] + 2 * _np.real(logψ_η_b[i]))
            )
        ).sum()
        low_range = s


# For expectation values of lindblad
def _local_values_purified_impl(op, machine, v, logψ, out):

    hi = machine.hilbert
    Np = hi.size_physical
    Na = hi.size_ancilla

    # Split tuple input
    σ_a, η_b = v
    logψ_σ_a, logψ_η_b = logψ

    assert σ_a.shape == η_b.shape

    # Extract the /sigma part of the state (the physical)
    σ = _np.asarray(σ_a[:, 0:Np])
    η = _np.asarray(η_b[:, 0:Np])

    # Extract the ancilla part of the state
    a = σ_a[:, Np : Np + Na]
    b = η_b[:, Np : Np + Na]

    sections = _np.empty(σ.shape[0], dtype=_np.int32)
    σ_primes, η_primes, mels = op._get_conn_flattened_(
        σ, η, sections, pad=False, merge_output=False
    )

    # sections are indices like in a sparse matrix and they don't have a leading 0.
    # Generate the length of each section
    section_lengths = _np.diff(sections, prepend=0)

    # Repeat every ancilla a sufficient number of times
    a_ext = _np.repeat(a, section_lengths, axis=0)
    b_ext = _np.repeat(b, section_lengths, axis=0)

    # join ancillas and states
    # σ_primes_b = _np.concatenate((σ_primes, b_ext), axis=1)
    # η_primes_b = _np.concatenate((η_primes, b_ext), axis=1)

    logψ_σ_primes_b = machine.log_val_ancilla(σ_primes, b_ext)
    logψ_η_primes_b = machine.log_val_ancilla(η_primes, b_ext)
    logψ_η_a = machine.log_val_ancilla(η, a)

    _local_values_purified_kernel(
        _np.asarray(logψ_σ_a),
        _np.asarray(logψ_η_b),
        _np.asarray(logψ_σ_primes_b),
        _np.asarray(logψ_η_primes_b),
        _np.asarray(logψ_η_a),
        mels,
        sections,
        out,
    )


@jit(nopython=True)
def _op_op_unpack_kernel(v, sections, vold):

    low_range = 0
    for i, s in enumerate(sections):
        vold[low_range:s] = v[i]
        low_range = s

    return vold


def _local_values_op_op_impl(op, machine, v, log_vals, out):

    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_np = _np.asarray(v)
    v_primes, mels = op.get_conn_flattened(v_np, sections)

    vold = _np.empty((sections[-1], v.shape[1]))
    _op_op_unpack_kernel(v_np, sections, vold)

    log_val_primes = machine.log_val(v_primes, vold)

    _local_values_kernel(
        _np.asarray(log_vals), _np.asarray(log_val_primes), mels, sections, out
    )


def local_values(op, machine, v, log_vals=None, out=None):
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

    # True when this is the local_value of a densitymatrix times an operator (observable)
    is_op_times_op = isinstance(machine, DensityMatrix) and not isinstance(
        op, _LocalLiouvillian
    )
    is_purified_times_op = isinstance(machine, PurifiedJaxMachine) and not isinstance(
        op, _LocalLiouvillian
    )
    is_purified_times_lind = isinstance(machine, PurifiedJaxMachine) and isinstance(
        op, _LocalLiouvillian
    )

    if log_vals is None:
        if is_op_times_op:
            log_vals = machine.log_val(v, v)
        elif is_purified_times_lind:
            log_vals = (machine.log_val(v[0]), machine.log_val(v[1]))
        else:
            log_vals = machine.log_val(v)

    if is_op_times_op:
        _impl = _local_values_op_op_impl
    elif is_purified_times_op:
        _impl = _local_values_purified_op_impl
    elif is_purified_times_lind:
        _impl = _local_values_purified_impl
    else:
        _impl = _local_values_impl

    if is_purified_times_op:
        assert (
            v.shape[1] == op.hilbert.size + machine.hilbert.size_ancilla
        ), "samples has wrong shape: {}; expected (?, {})".format(
            v.shape, op.hilbert.size
        )
        if v.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

    elif is_purified_times_lind:
        assert len(v) == 2, "Purified samples should be a tuple of row and columns"
        assert (
            v[0].shape == v[1].shape
        ), "Purified samples should have the same shape but they have {} and {}".format(
            v[0].shape, v[1].shape
        )
        assert (
            v[0].shape[1] == op.hilbert.size_physical + machine.hilbert.size_ancilla
        ), "samples has wrong shape: {}; expected (?, {} + {})".format(
            v[0].shape, op.hilbert.size_physical, machine.hilbert.size_ancilla
        )
    else:
        assert (
            v.shape[1] == op.hilbert.size
        ), "samples has wrong shape: {}; expected (?, {})".format(
            v.shape, op.hilbert.size
        )
        if v.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

    if out is None:
        if is_purified_times_lind:
            Nb = v[0].shape[0]
        else:
            Nb = v.shape[0]

        out = _np.empty(Nb, dtype=_np.complex128)

    _impl(op, machine, v, log_vals, out)

    return out
