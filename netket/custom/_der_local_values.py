import numpy as _np
from numba import jit


@jit(nopython=True)
def _der_local_values_kernel(log_vals, log_val_primes, der_log_val_primes, mels, sections, out, der_out):
    low_range = 0
    out.fill(0.0)
    der_out.fill(0.0)
    for i, s in enumerate(sections):
        for j in range(low_range, s):
            val = mels[j] * _np.exp(log_val_primes[j] - log_vals[i])
            out[i] += val
            der_out[i, :] += val * der_log_val_primes[j, :]
        low_range = s


def _der_local_values(op, machine, v, log_vals, out, der_out):

    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_primes, mels = op.get_conn_flattened(_np.asarray(v), sections)

    log_val_primes = machine.log_val(v_primes)
    der_log_val_primes = machine.der_log(v_primes)


    _der_local_values_kernel(
        _np.asarray(log_vals), _np.asarray(log_val_primes), _np.asarray(der_log_val_primes), mels, sections, out, der_out
    )

def local_values_with_der(op, machine, v, log_vals=None, out=None, der_out=None):
    if v.ndim != 2:
        raise RuntimeError("Invalid input shape, expected a 2d array")

    if log_vals is None:
        log_vals = machine.log_val(v)

    assert (
        v.shape[1] == op.hilbert.size
    ), "samples has wrong shape: {}; expected (?, {})".format(v.shape, op.hilbert.size)

    if out is None:
        out = _np.empty(v.shape[0], dtype=_np.complex128)
    
    if der_out is None:
        der_out = _np.empty((v.shape[0], machine.parameters.shape[0]), dtype=_np.complex128)

    _der_local_values(op, machine, v, log_vals, out, der_out)

    return (out, der_out)
