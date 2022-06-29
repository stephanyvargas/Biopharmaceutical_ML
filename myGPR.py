from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.optimize import _check_optimize_result

from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize

class MyGPR(GaussianProcessRegressor):
    def __init__(self, max_iter=15000, kernel=None, *,
                 alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super().__init__(kernel=None,
                     alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                     normalize_y=False, copy_X_train=True, random_state=None)
        self._max_iter = max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        def new_optimizer(obj_func, initial_theta, bounds):
            return scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                max_iter=self._max_iter,
            )
        self.optimizer = new_optimizer
        return super()._constrained_optimization(obj_func, initial_theta, bounds)

class MyGPR_2(GaussianProcessRegressor):
    def __init__(self, max_iter=2e05, gtol=1e-06, kernel=None, *,
                 alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super().__init__(kernel=None,
                     alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                     normalize_y=False, copy_X_train=True, random_state=None)
        self._max_iter = max_iter
        self._gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta,
                            method="L-BFGS-B", jac=True, bounds=bounds,
                            options={'maxiter':self._max_iter, 'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min

#lbfgs
