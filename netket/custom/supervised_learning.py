import numpy as np
from numba import njit, prange
import scipy as sp

class SupervisedLearning():
    def __init__(self, machine):
        self.machine = machine
    
    def mean_squared_error(self, basis, target_amplitudes, weightings):
        return np.sum(weightings * abs(np.exp(self.machine.log_val(basis)) - target_amplitudes)**2)

    def mean_squared_error_der(self, basis, target_amplitudes, weightings):
        estimates = np.exp(self.machine.log_val(basis))
        der_log = self.machine.der_log(basis)
        residuals = (estimates - target_amplitudes).conj()*weightings
        der = 2 * np.einsum("ij,i,i->j", der_log, estimates, residuals)
        if self.machine.has_complex_parameters:
            der = np.concatenate((der, 1.j*der))
        return der.real

    def mean_squared_error_hess(self, basis, target_amplitudes, weightings):
        estimates = np.exp(self.machine.log_val(basis))
        der = self.machine.der_log(basis)
        proper_der = (der.T * estimates)
        hess_el = self.machine.hess(basis)
        wfn_hess_first_term = hess_el.T * estimates
        wfn_hess_sec_term = np.einsum("ij,jk->ikj", proper_der, der)
        wfn_hess = wfn_hess_first_term + wfn_hess_sec_term
        residuals = (estimates-target_amplitudes).conj()*weightings
        hess_first_term = (np.dot(wfn_hess, residuals))
        hess_sec_term = np.matmul(proper_der*weightings, proper_der.T.conj())
        if self.machine.has_complex_parameters:
            hess_first_term = np.block([[hess_first_term, 1.j*hess_first_term],[1.j*hess_first_term,-hess_first_term]])
            hess_sec_term = np.block([[hess_sec_term, -1.j*hess_sec_term],[1.j*hess_sec_term, hess_sec_term]])
        return (2 * (hess_first_term + hess_sec_term).real)

    def log_overlap(self, basis, target_amplitudes, weightings):
        predictions = np.exp(self.machine.log_val(basis))
        overlap = abs(np.sum(weightings * (predictions * target_amplitudes.conj())))**2
        norm = np.sum(weightings * abs(predictions)**2) * np.sum(weightings * abs(target_amplitudes)**2)
        return -np.log(overlap/norm)

    def log_overlap_der(self, basis, target_amplitudes, weightings):
        estimates = np.exp(self.machine.log_val(basis)).conj()
        der = self.machine.der_log(basis).conj()
        estimates_tiled = np.tile(estimates, (der.shape[1], 1)).T
        target_amps_tiled = np.tile(target_amplitudes, (der.shape[1], 1)).T
        weightings_tiled = np.tile(weightings, (der.shape[1], 1)).T
        overlap1 = np.sum(weightings_tiled * estimates_tiled * der * target_amps_tiled, axis=0)
        norm1 = np.sum(weightings * estimates * target_amplitudes)
        overlap2 = np.sum(weightings_tiled * (abs(estimates_tiled)**2 * der), axis=0)
        norm2 = np.sum(weightings * abs(estimates)**2)
        derivative = -overlap1/norm1 + overlap2/norm2
        if self.machine.has_complex_parameters:
            derivative = np.concatenate((derivative.real, derivative.imag))
        return derivative.real

    def bayes_loss(self, basis, target_amplitudes, weightings, beta, alpha):
        parameters = self.machine.parameters
        if self.machine.has_complex_parameters:
            parameters = np.concatenate((parameters.real, parameters.imag))
        return beta/2 * self.mean_squared_error(basis, target_amplitudes, weightings) + 0.5 * np.sum((parameters**2) * alpha)

    def grad_bayes(self, basis, target_amplitudes, weightings, beta, alpha):
        parameters = self.machine.parameters
        if self.machine.has_complex_parameters:
            parameters = np.concatenate((parameters.real, parameters.imag))
        der = beta/2 * self.mean_squared_error_der(basis, target_amplitudes, weightings)
        der += parameters * alpha
        return der

    def hess_bayes(self, basis, target_amplitudes, weightings, beta, alpha):
        parameters = self.machine.parameters
        if self.machine.has_complex_parameters:
            parameters = np.concatenate((parameters.real, parameters.imag))
        hess = beta/2 * self.mean_squared_error_hess(basis, target_amplitudes, weightings)
        hess += np.diag(alpha)
        return hess

    def get_bias(self, target_amplitudes, weightings=None, dtype=complex):
        if weightings is None:
            return np.mean(np.log(target_amplitudes))
        else:
            return np.sum(np.log(target_amplitudes)*weightings)/np.sum(weightings)

class QGPSLearning(SupervisedLearning):
    @staticmethod
    @njit(parallel=True)
    def kernel_mat_inner(epsilon, ref_site, confs, K, complex_par, Smap, Smap_inverse, sym_spin_flip_sign):
        for i in prange(confs.shape[0]):
            for x in range(epsilon.shape[1]):
                for t in range(Smap.shape[0]): 
                    innerprod = 1.0
                    if complex_par:
                        innerprod = np.complex128(1.0)
                    for j in range(confs.shape[1]):
                        if j != ref_site:
                            if sym_spin_flip_sign[t] * confs[i, Smap[t,j]] < 0:
                                innerprod *= epsilon[j, x, 0]
                            else:
                                innerprod *= epsilon[j, x, 1]
                    if sym_spin_flip_sign[t] * confs[i, Smap[t, ref_site]] < 0.0:
                        K[i, 2*x] += innerprod
                    else:
                        K[i, 2*x+1] += innerprod
        return K

    def kernel_mat(self, ref_site, confs, K=None):
        if K is None:
            K = np.zeros((confs.shape[0], self.machine._epsilon.shape[1] * 2), dtype=self.machine._epsilon.dtype)
        else:
            K.fill(0.0)
        if K.dtype == complex:
            complex_par = True
        else:
            complex_par = False
        return self.kernel_mat_inner(self.machine._epsilon, ref_site, confs, K, complex_par, self.machine._Smap, self.machine._Smap_inverse, self.machine._sym_spin_flip_sign)

    def log_marg_lik(self, confset, target_amplitudes, K, noise_tilde=1.e-2, alpha=None):
        n_bond = self.machine._epsilon.shape[1]
        n_sites = self.machine._epsilon.shape[0]
        if alpha is None:
            alpha_mat = np.zeros(2*n_bond)
        elif type(alpha) == np.ndarray:
            alpha_mat = alpha[:].copy()
        else:
            alpha_mat = alpha * np.ones(2*n_bond)
        t = target_amplitudes.astype(self.machine._epsilon.dtype)
        if noise_tilde == 0.:
            S_diag = np.ones(len(t))
        else:
            S_diag = 1/(np.log1p(noise_tilde/(abs(t)**2)))
        fit_data = np.log(t)
        KtK = np.dot(K.conj().T, np.dot(np.diag(S_diag), K))
        KtK += np.diag(alpha_mat[ :])
        y = K.conj().T.dot(S_diag * fit_data)
        log_lik = np.sum(np.log(alpha_mat[:]))
        try:
            L = sp.linalg.cholesky(KtK, lower=True)
            weights = sp.linalg.cho_solve((L, True), y)
            Lprime = sp.linalg.solve_triangular(L, np.eye(KtK.shape[0]), check_finite=False, lower=True)
            log_lik += 2*np.sum(np.log(np.diag(Lprime)))
        except:
            L = np.linalg.inv(KtK)
            weights = L.dot(y)
            log_lik += np.linalg.slogdet(L)[1]
        log_lik -= np.sum(np.log(2*np.pi*1/S_diag))
        log_lik -= np.dot(fit_data.conj(), S_diag * fit_data)
        log_lik += np.dot(weights.conj(), np.dot(KtK, weights))
        return 0.5*log_lik.real

    def log_marg_lik_noise_der(self, confset, target_amplitudes, K, noise_tilde=1.e-2, alpha=None):
        n_bond = self.machine._epsilon.shape[1]
        n_sites = self.machine._epsilon.shape[0]
        if alpha is None:
            alpha_mat = np.zeros(2*n_bond)
        elif type(alpha) == np.ndarray:
            alpha_mat = alpha[:].copy()
        else:
            alpha_mat = alpha * np.ones(2*n_bond)
        t = target_amplitudes.astype(self.machine._epsilon.dtype)
        if noise_tilde == 0.:
            S_diag = np.ones(len(t))
        else:
            S_diag = 1/(np.log1p(noise_tilde/(abs(t)**2)))
        fit_data = np.log(t)
        KtK = np.dot(K.conj().T, np.dot(np.diag(S_diag), K))
        KtK += np.diag(alpha_mat[ :])
        y = K.conj().T.dot(S_diag * fit_data)
        del_S = 1/((abs(t)**2) * (1 + noise_tilde/(abs(t)**2)))
        derivative_noise = -np.sum(S_diag * del_S)
        Delta_S = - (S_diag**2 * del_S)
        try:
            L = sp.linalg.cholesky(KtK, lower=True)
            weights = sp.linalg.cho_solve((L, True), y)
            Sinv = sp.linalg.solve_triangular(L, np.eye(KtK.shape[0]), check_finite=False, lower=True)
            derivative_noise -= np.trace(K.conj().T.dot(np.diag(Delta_S).dot(K.dot(Sinv.T.dot(Sinv)))))
        except:
            print("Warning, cholesky failed!")
            Sinv = np.linalg.inv(KtK)
            weights = Sinv.dot(y)
            derivative_noise -= np.trace(K.conj().T.dot(np.diag(Delta_S).dot(K.dot(Sinv))))
        derivative_noise -= fit_data.conj().dot(Delta_S*fit_data)
        derivative_noise -= weights.conj().dot(K.conj().T.dot(Delta_S*K.dot(weights)))
        derivative_noise += 2*fit_data.conj().dot(Delta_S*K.dot(weights))
        return 0.5*derivative_noise.real
    
    def log_marg_lik_alpha_der(self, confset, target_amplitudes, K, noise_tilde=1.e-2, alpha=None):
        n_bond = self.machine._epsilon.shape[1]
        n_sites = self.machine._epsilon.shape[0]
        if alpha is None:
            alpha_mat = np.zeros(2*n_bond)
        elif type(alpha) == np.ndarray:
            alpha_mat = alpha[:].copy()
        else:
            alpha_mat = alpha * np.ones(2*n_bond)
        t = target_amplitudes.astype(self.machine._epsilon.dtype)
        if noise_tilde == 0.:
            S_diag = np.ones(len(t))
        else:
            S_diag = 1/(np.log1p(noise_tilde/(abs(t)**2)))
        fit_data = np.log(t)
        KtK = np.dot(K.conj().T, np.dot(np.diag(S_diag), K))
        KtK += np.diag(alpha_mat[ :])
        y = K.conj().T.dot(S_diag * fit_data)
        derivative_alpha = 1/alpha_mat
        try:
            L = sp.linalg.cholesky(KtK, lower=True)
            weights = sp.linalg.cho_solve((L, True), y)
            Sinv = sp.linalg.solve_triangular(L, np.eye(KtK.shape[0]), check_finite=False, lower=True)
            derivative_alpha -= np.sum(abs(Sinv) ** 2, 0)
        except:
            print("Warning, cholesky failed!")
            Sinv = np.linalg.inv(KtK)
            weights = Sinv.dot(y)
            derivative_alpha -= np.diag(Sinv).real
        derivative_alpha -= (weights.conj() * weights).real
        return 0.5*derivative_alpha.real

    def fit_step(self, confset, target_amplitudes, ref_site, noise_tilde, alpha, noise_range=(1.e-10, 1.e5), alpha_range=(1.e-10, 1.e7),
                 opt_alpha=True, opt_noise=True, max_iterations=20, rvm=False):
        n_bond = self.machine._epsilon.shape[1]
        K = self.kernel_mat(ref_site, confset)
        alpha_mat = alpha[ref_site, :].copy()
        if opt_noise:
            def derivative(x):
                der_noise = self.log_marg_lik_noise_der(confset, target_amplitudes, K, noise_tilde=np.exp(x[0]), alpha=np.exp(x[1]))
                if opt_alpha:
                    if rvm:
                        der_alpha = self.log_marg_lik_alpha_der(confset, target_amplitudes, K, noise_tilde=np.exp(x[0]), alpha=np.exp(x[1:]))
                        return np.concatenate((np.array([-der_noise]), -der_alpha))*np.exp(x)
                    else:
                        der_alpha = self.log_marg_lik_alpha_der(confset, target_amplitudes, K, noise_tilde=np.exp(x[0]), alpha=np.exp(x[1]))
                        return np.array([-der_noise, -np.sum(der_alpha)])*np.exp(x)
                else:
                    return - der_noise * np.exp(x)
            if opt_alpha:
                if rvm:
                    bounds = [(np.log(noise_range[0]), np.log(noise_range[1]))]
                    for i in range(alpha.shape[1]):
                        bounds.append((np.log(alpha_range[0]), np.log(alpha_range[1])))

                    ML = lambda x : -self.log_marg_lik(confset, target_amplitudes, K, noise_tilde=np.exp(x[0]), alpha=np.exp(x[1:]))
                    opt = sp.optimize.minimize(ML, np.concatenate((np.array([np.log(noise_tilde[ref_site])]), np.log(alpha_mat))), options={"maxiter" : max_iterations}, jac=derivative, bounds=bounds)
                    new_hyperpar = np.exp(opt.x)
                    noise_tilde[ref_site] = new_hyperpar[0]
                    np.copyto(alpha_mat, new_hyperpar[1:])
                else:
                    ML = lambda x : -self.log_marg_lik(confset, target_amplitudes, K, noise_tilde=np.exp(x[0]), alpha=np.exp(x[1]))
                    opt = sp.optimize.minimize(ML, np.array([np.log(noise_tilde[ref_site]), np.log(alpha_mat[0])]), options={"maxiter" : max_iterations}, jac=derivative, bounds=[(np.log(noise_range[0]), np.log(noise_range[1])), (np.log(alpha_range[0]), np.log(alpha_range[1]))])
                    new_hyperpar = np.exp(opt.x)
                    noise_tilde[ref_site] = new_hyperpar[0]
                    alpha_mat[:] = new_hyperpar[1]
            else:
                ML = lambda x : -self.log_marg_lik(confset, target_amplitudes, K, noise_tilde=np.exp(x[0]), alpha=alpha_mat)
                opt = sp.optimize.minimize(ML, np.log(noise_tilde[ref_site]), options={"maxiter" : max_iterations}, jac=derivative, bounds=[(np.log(noise_range[0]), np.log(noise_range[1]))])
                new_hyperpar = np.exp(opt.x)
                noise_tilde[ref_site] = new_hyperpar
        elif opt_alpha:
            noise = noise_tilde[ref_site]
            t = target_amplitudes.astype(self.machine._epsilon.dtype)
            if noise == 0.:
                S_diag = np.ones(len(t))
            else:
                S_diag = 1/(np.log1p(noise/(abs(t)**2)))
            fit_data = np.log(t)
            KtK = np.dot(K.conj().T, np.dot(np.diag(S_diag), K))
            for i in range(max_iterations):
                active_elements = alpha_mat < alpha_range[1]
                y = (K.conj().T.dot(S_diag * fit_data))[active_elements]
                KtK_alph = (KtK + np.diag(alpha_mat[ :]))[np.ix_(active_elements, active_elements)]
                try:
                    L = sp.linalg.cholesky(KtK_alph, lower=True)
                    weights = sp.linalg.cho_solve((L, True), y)
                    Sinv = sp.linalg.solve_triangular(L, np.eye(KtK_alph.shape[0]), check_finite=False, lower=True)
                    gamma = (1 - (alpha_mat[active_elements])*np.sum(abs(Sinv) ** 2, 0))
                except:
                    print("Warning, cholesky failed!")
                    Sinv = np.linalg.inv(KtK_alph)
                    weights = Sinv.dot(y)
                    gamma = (1 - (alpha_mat[active_elements])*np.diag(Sinv).real)
                if rvm:
                    alpha_mat[active_elements] = (gamma/(weights.conj()*weights)).real
                else:
                    alpha_mat.fill((np.sum(gamma)/(weights.conj().dot(weights))).real)

        np.copyto(alpha[ref_site,:], alpha_mat)

        noise = noise_tilde[ref_site]
        t = target_amplitudes.astype(self.machine._epsilon.dtype)
        if noise == 0.:
            S_diag = np.ones(len(t))
        else:
            S_diag = 1/(np.log1p(noise/(abs(t)**2)))
        fit_data = np.log(t)
        KtK = np.dot(K.conj().T, np.dot(np.diag(S_diag), K))
        active_elements = alpha_mat < alpha_range[1]
        KtK_alph = (KtK + np.diag(alpha_mat[ :]))[np.ix_(active_elements, active_elements)]
        y = (K.conj().T.dot(S_diag * fit_data))[active_elements]
        try:
            L = sp.linalg.cholesky(KtK_alph, lower=True)
            weights = sp.linalg.cho_solve((L, True), y)
        except:
            print("Warning, cholesky failed2!")
            weights = np.linalg.lstsq(KtK_alph, y)[0]
        full_weights = np.zeros(KtK.shape[0], dtype=weights.dtype)
        full_weights[active_elements] = weights
        self.machine._epsilon[ref_site, :, :] = full_weights.reshape(n_bond, 2)
        return
