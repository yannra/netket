import numpy as np

def get_hash(ma, basisconfig):
    hash_val = hash(tuple(basisconfig))
    shifted_conf = np.zeros(ma._Smap.shape[1])
    for i in range(ma._Smap.shape[0]):
        for j in range(ma._Smap.shape[1]):
            shifted_conf[j] = ma._sym_spin_flip_sign[i] * basisconfig[ma._Smap[i,j]]
        new_hash = hash(tuple(shifted_conf))
        if new_hash < hash_val:
            hash_val = new_hash
    return hash_val

def get_symmetric_inequivalent_set(ma, basis, amplitudes):
    hash_dict = {}
    for (i, k) in enumerate(basis):
        hash_val = get_hash(ma, k)
        if hash_val not in hash_dict:
            hash_dict[hash_val] = [1, i]
        else:
            hash_dict[hash_val][0] += 1
            if abs(amplitudes[hash_dict[hash_val][1]] - amplitudes[i]) > 1.e-6:
                print(amplitudes[i], amplitudes[hash_dict[hash_val][1]], basis[i], basis[hash_dict[hash_val][1]])
    pruned_basis = []
    pruned_amplitudes = []
    weightings = []
    for key in hash_dict.keys():
        pruned_basis.append(basis[hash_dict[key][1]])
        pruned_amplitudes.append(amplitudes[hash_dict[key][1]])
        weightings.append(hash_dict[key][0])
    return (np.array(pruned_basis), np.array(pruned_amplitudes), np.array(weightings))

def evaluate_exact_energy(ma, ha, states):
    amps = np.exp(ma.log_val(states))
    return amps.dot(ha.dot(amps.conj()))/(amps.dot(amps.conj()))