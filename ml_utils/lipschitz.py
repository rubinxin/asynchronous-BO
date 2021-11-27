from typing import Optional, Dict

import numpy as np
from scipy import optimize as spo

from ml_utils.models import GP
from ml_utils.optimization import minimize_with_restarts


def estimate_lipschitz_constant(
        surrogate: GP,
        bounds: Optional[np.ndarray] = None,
        num_restarts: Optional[int] = 10,
        minimize_options: Optional[Dict] = {'maxiter': 200}) -> float:
    """
    Estimates the Lipschitz constant of the surrogate

    Returns
    -------
    float
        Lipschitz constant estimate
    """

    def negative_df(x: np.ndarray) -> np.ndarray:
        model = surrogate
        x = np.atleast_2d(x)
        dmdx = model.dmu_dx(x).sum(-1)
        # simply take the norm of the expectation of the gradient
        res = np.sqrt((dmdx * dmdx).sum(1))
        return -res

    if bounds is None:
        # TODO: test this
        # No bounds, no restarts, so start at highest grad in surrogate data
        idx_biggest_grad = np.argmax(surrogate.dmu_dx(surrogate.X).sum(-1))
        opt_result = spo.minimize(negative_df, surrogate.X[idx_biggest_grad])
    else:
        opt_result = minimize_with_restarts(negative_df, bounds,
                                            num_restarts=num_restarts,
                                            hard_bounds=bounds,
                                            minimize_options=minimize_options)
    best_negative_df = opt_result.fun.item()
    L = -best_negative_df

    # to avoid problems in cases in which the model is flat.
    if L < 1e-7:
        L = 10

    return L


def estimate_lipschitz_around_x(x: np.ndarray, surrogate, bounds) -> float:
    """Find the Lipschitz constant in a region close to x

    Parameters
    ----------
    x

    Returns
    -------
    L
    """
    # Search spaces for the local Lipschitz constant optimization
    theta = surrogate.kern.lengthscale
    lower_sp = np.maximum(bounds[:, 0], x - theta)
    upper_sp = np.minimum(bounds[:, 1], x + theta)
    lower_sp = lower_sp.reshape(-1, 1)
    upper_sp = upper_sp.reshape(-1, 1)
    lipschitz_search_space = np.hstack((lower_sp, upper_sp))
    L = estimate_lipschitz_constant(surrogate,
                                    lipschitz_search_space)
    return L