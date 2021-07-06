import numpy as np
from numba import njit
import scipy as sp

from netket.stats import (
    statistics as _statistics,
    mean as _mean,
    sum_inplace as _sum_inplace,
)

from netket.utils import (
    MPI_comm as _MPI_comm,
    n_nodes as _n_nodes,
    node_number as _rank
)

class SupervisedLearning():
    def __init__(self, machine):
        self.machine = machine
    
    def mean_squared_error(self, basis, target_amplitudes, weightings):
        return _MPI_comm.allreduce(np.sum(weightings * abs(np.exp(self.machine.log_val(basis)) - target_amplitudes)**2))

    def mean_squared_error_der(self, basis, target_amplitudes, weightings):
        estimates = np.exp(self.machine.log_val(basis))
        der_log = self.machine.der_log(basis)
        residuals = (estimates - target_amplitudes).conj()*weightings
        der = 2 * np.einsum("ij,i,i->j", der_log, estimates, residuals)
        if self.machine.has_complex_parameters:
            der = np.concatenate((der, 1.j*der))
        return _sum_inplace(der.real)

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
        return _sum_inplace(2 * (hess_first_term + hess_sec_term).real)

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
            return _mean(np.log(target_amplitudes))
        else:
            return _MPI_comm.allreduce(np.sum(np.log(target_amplitudes)*weightings))/_MPI_comm.allreduce(np.sum(weightings))

class QGPSLearning(SupervisedLearning):
    def __init__(self, machine, init_alpha = 1.0, init_noise_tilde = 1.e-1):
        super().__init__(machine)
        self.K = None
        self.weights = None
        self.noise_tilde = init_noise_tilde
        self.alpha_mat = np.ones((self.machine._epsilon.shape[0], 2*self.machine._epsilon.shape[1]))*init_alpha
        self.alpha_cutoff = 1.e8
        self.site_prod = None
        self.confs = None
        self.ref_site = None
        assert(init_alpha < self.alpha_cutoff)

    @staticmethod
    @njit()
    def kernel_mat_inner(site_prod, ref_site, confs, Smap, sym_spin_flip_sign, K):
        K.fill(0.0)
        for i in range(site_prod.shape[0]):
            for x in range(site_prod.shape[1]):
                for t in range(site_prod.shape[2]): 
                    if sym_spin_flip_sign[t] * confs[i, Smap[t, ref_site]] < 0.0:
                        K[i, 2*x] += site_prod[i, x, t]
                    else:
                        K[i, 2*x+1] += site_prod[i, x, t]
        return K

    @staticmethod
    @njit()
    def compute_site_prod_fast(epsilon, ref_site, confs, Smap, sym_spin_flip_sign, site_product):
        site_product.fill(1.0)
        for i in range(confs.shape[0]):
            for x in range(epsilon.shape[1]):
                for t in range(Smap.shape[0]): 
                    for j in range(confs.shape[1]):
                        if j != ref_site:
                            if sym_spin_flip_sign[t] * confs[i, Smap[t,j]] < 0:
                                site_product[i, x, t] *= epsilon[j, x, 0]
                            else:
                                site_product[i, x, t] *= epsilon[j, x, 1]
        return site_product
    
    @staticmethod
    @njit()
    def update_site_prod_fast(epsilon, ref_site, ref_site_old, confs, Smap, sym_spin_flip_sign, site_product):
        eps = np.finfo(np.double).eps
        for x in range(epsilon.shape[1]):
            if abs(epsilon[ref_site, x, 0]) > 1.e4 * eps and abs(epsilon[ref_site, x, 0]) > 1.e4 * eps:
                for i in range(confs.shape[0]):
                    for t in range(Smap.shape[0]):
                        if sym_spin_flip_sign[t] * confs[i, Smap[t,ref_site]] < 0:
                            site_product[i, x, t] /= epsilon[ref_site, x, 0]
                        else:
                            site_product[i, x, t] /= epsilon[ref_site, x, 1]

                        if sym_spin_flip_sign[t] * confs[i, Smap[t,ref_site_old]] < 0:
                            site_product[i, x, t] *= epsilon[ref_site_old, x, 0]
                        else:
                            site_product[i, x, t] *= epsilon[ref_site_old, x, 1]
            else:
                for i in range(confs.shape[0]):
                    for t in range(Smap.shape[0]):
                        site_product[i, x, t] = 1.0
                        for j in range(confs.shape[1]):
                            if j != ref_site:
                                if sym_spin_flip_sign[t] * confs[i, Smap[t,j]] < 0:
                                    site_product[i, x, t] *= epsilon[j, x, 0]
                                else:
                                    site_product[i, x, t] *= epsilon[j, x, 1]

        return site_product

    def compute_site_prod(self):
        if self.site_prod is None:
            self.site_prod = np.zeros((self.confs.shape[0], self.machine._epsilon.shape[1], self.machine._Smap.shape[0]), dtype=self.machine._epsilon.dtype)
        self.site_prod = self.compute_site_prod_fast(self.machine._epsilon, self.ref_site, self.confs, self.machine._Smap,
                                                     self.machine._sym_spin_flip_sign, self.site_prod)
        self.site_prod_ref_site = self.ref_site

    def update_site_prod(self):
        if self.site_prod_ref_site != self.ref_site:
            self.site_prod = self.update_site_prod_fast(self.machine._epsilon, self.ref_site, self.site_prod_ref_site, self.confs, self.machine._Smap,
                                                        self.machine._sym_spin_flip_sign, self.site_prod)
        self.site_prod_ref_site = self.ref_site
    

    def set_kernel_mat(self, confs):
        if self.site_prod is None or self.confs is None or np.sum(self.confs != confs) != 0:
            self.confs = confs
            self.compute_site_prod()

        elif self.ref_site != self.site_prod_ref_site:
            self.update_site_prod()

        if self.K is None:
            self.K = np.zeros((confs.shape[0], self.machine._epsilon.shape[1] * 2), dtype=self.machine._epsilon.dtype)

        self.K = self.kernel_mat_inner(self.site_prod, self.ref_site, self.confs, self.machine._Smap,
                                       self.machine._sym_spin_flip_sign, self.K)
        return self.K

    def reset(self):
        self.site_prod = None

    def setup_fit_alpha_dep(self, confset, target_amplitudes):
        self.active_elements = self.alpha_mat[self.ref_site,:] < self.alpha_cutoff
        self.KtK_alpha = self.KtK + np.diag(self.alpha_mat[self.ref_site,:])
        
        try:
            L = sp.linalg.cholesky(self.KtK_alpha[np.ix_(self.active_elements, self.active_elements)], lower=True)
            self.Sinv = sp.linalg.solve_triangular(L, np.eye(self.KtK_alpha.shape[0]), check_finite=False, lower=True)
            weights = sp.linalg.cho_solve((L, True), self.y[self.active_elements])
            self.cholesky = True
        except:
            self.Sinv = np.linalg.inv(self.KtK_alpha[np.ix_(self.active_elements, self.active_elements)])
            weights = self.Sinv.dot(self.y[self.active_elements])
            self.cholesky = False
            # if _rank == 0:
            #     print("Warning! Cholesky failed.")

        if self.weights is None:
            self.weights = np.zeros(self.alpha_mat.shape[1], dtype=weights.dtype)
        
        # potentially distribute weights across processes
        self.weights[self.active_elements] = weights

    def setup_fit_noise_dep(self, confset, target_amplitudes):
        if self.noise_tilde == 0.:
            self.S_diag = np.ones(len(self.log_amps))
        else:
            self.S_diag = 1/(np.log1p(self.noise_tilde/(abs(self.log_amps)**2)))

        self.KtK = _sum_inplace(np.dot(self.K.conj().T, np.dot(np.diag(self.S_diag), self.K)))
        self.y = _sum_inplace(self.K.conj().T.dot(self.S_diag * self.fit_data))

        self.setup_fit_alpha_dep(confset, target_amplitudes)

    def setup_fit(self, confset, target_amplitudes, ref_site):
        self.ref_site = ref_site
        self.log_amps = target_amplitudes.astype(self.machine._epsilon.dtype)
        self.fit_data = np.log(self.log_amps)
        self.set_kernel_mat(confset)
        self.setup_fit_noise_dep(confset, target_amplitudes)

    def log_marg_lik(self):
        log_lik = -(np.sum(np.log(2*np.pi*1/self.S_diag)))
        log_lik -= np.dot(self.fit_data.conj(), self.S_diag * self.fit_data)
        log_lik = _MPI_comm.allreduce(log_lik)

        if self.cholesky:
            log_lik += 2*np.sum(np.log(np.diag(self.Sinv)))
        else:
            log_lik += np.linalg.slogdet(self.Sinv)[1]

        log_lik += np.sum(np.log(self.alpha_mat[self.ref_site, :]))
        weights = self.weights[self.active_elements]
        log_lik += np.dot(weights.conj(), np.dot(self.KtK_alpha[np.ix_(self.active_elements, self.active_elements)], weights))
        return 0.5*log_lik.real

    def log_marg_lik_noise_der(self):
        del_S = 1/((abs(self.log_amps)**2) * (1 + self.noise_tilde/(abs(self.log_amps)**2)))
        Delta_S = - (self.S_diag**2 * del_S)

        derivative_noise = -np.sum(self.S_diag * del_S)

        K = self.K[:,self.active_elements]
        if self.cholesky:
            derivative_noise -= np.trace(K.conj().T.dot(np.diag(Delta_S).dot(K.dot(self.Sinv.T.dot(self.Sinv)))))
        else:
            derivative_noise -= np.trace(K.conj().T.dot(np.diag(Delta_S).dot(K.dot(self.Sinv))))

        weights = self.weights[self.active_elements]

        derivative_noise -= self.fit_data.conj().dot(Delta_S*self.fit_data)
        derivative_noise -= weights.conj().dot(K.conj().T.dot(Delta_S*K.dot(weights)))
        derivative_noise += 2*self.fit_data.conj().dot(Delta_S*K.dot(weights))

        derivative_noise = _MPI_comm.allreduce(derivative_noise)

        return 0.5*derivative_noise.real
    
    def log_marg_lik_alpha_der(self):
        derivative_alpha = 1/(self.alpha_mat[self.ref_site, :])
        if self.cholesky:
            derivative_alpha[self.active_elements] -= np.sum(abs(self.Sinv) ** 2, 0)
        else:
            derivative_alpha[self.active_elements] -= np.diag(self.Sinv).real

        derivative_alpha -= (self.weights.conj() * self.weights).real

        return 0.5*derivative_alpha.real
    
    def opt_alpha(self, confset, target_amplitudes, iterations=5, rvm=True):
        for i in range(iterations):
            if self.cholesky:
                gamma = (1 - (self.alpha_mat[self.ref_site, self.active_elements])*np.sum(abs(self.Sinv) ** 2, 0))
            else:
                gamma = (1 - (self.alpha_mat[self.ref_site, self.active_elements])*np.diag(self.Sinv).real)
            if rvm:
                self.alpha_mat[self.ref_site, self.active_elements] = (gamma/((self.weights.conj()*self.weights)[self.active_elements])).real
            else:
                self.alpha_mat[self.ref_site, :] = ((np.sum(gamma)/(self.weights.conj().dot(self.weights))).real)
            self.setup_fit_alpha_dep(confset, target_amplitudes)


    def fit_step(self, confset, target_amplitudes, ref_site, noise_range=(1.e-10, 1.e5),
                 opt_alpha=True, opt_noise=True, alpha_iterations=5, noise_iterations=1, rvm=False):
        self.setup_fit(confset, target_amplitudes, ref_site)
        if opt_noise:
            alpha_init = self.alpha_mat.copy()
            def ML(x):
                self.noise_tilde = np.exp(x)
                if opt_alpha:
                    np.copyto(self.alpha_mat, alpha_init)
                self.setup_fit_noise_dep(confset, target_amplitudes)
                if opt_alpha:
                    self.opt_alpha(confset, target_amplitudes, iterations=alpha_iterations, rvm=rvm)
                return -self.log_marg_lik()

            def derivative(x):
                self.noise_tilde = np.exp(x)
                if opt_alpha:
                    np.copyto(self.alpha_mat, alpha_init)
                self.setup_fit_noise_dep(confset, target_amplitudes)
                if opt_alpha:
                    self.opt_alpha(confset, target_amplitudes, iterations=alpha_iterations, rvm=rvm)
                der_noise = self.log_marg_lik_noise_der()
                return - der_noise * np.exp(x)
            
            bounds = [(np.log(noise_range[0]), np.log(noise_range[1]))]

            opt = sp.optimize.minimize(ML, np.log(self.noise_tilde), options={"maxiter" : noise_iterations}, jac=derivative, bounds=bounds)
            self.noise_tilde = np.exp(opt.x)[0]
            if opt_alpha:
                np.copyto(self.alpha_mat, alpha_init)
            self.setup_fit_noise_dep(confset, target_amplitudes)

        if opt_alpha:
            self.opt_alpha(confset, target_amplitudes, iterations=alpha_iterations, rvm=rvm)

        self.machine._epsilon[ref_site, :, :] = self.weights.reshape(self.machine._epsilon.shape[1], 2)
        return
