# -*- coding: utf-8 -*-

"""
Test functions for global optimisation.

Some of these are based on the following library:
    BayesOpt, an efficient C++ library for Bayesian optimization.
"""
from typing import Tuple, Callable

import numpy as np
import GPy


def draw_from_a_gp(dim, x_lim, n=None, kern=None, seed=None):
    """
    A draw from a known GP kernel

    :dim (int): Dimensionality of x

    :x_lim (ndarray): limits of x

    :n (int): Total number of points used to generate the surface

    :kern (GPy.kern): kernel from which to draw the function

    Returns

    :f (func): Function object that provides the function value at a given x
    """
    assert dim == x_lim.shape[0]
    np.random.seed(seed)

    if n is None:
        n = 300

    n_for_each_dim = int(n ** (1 / dim))
    if kern is None:
        kern = GPy.kern.RBF(dim, lengthscale=0.4, variance=1.)

    single_x_vecs = []
    for ii in range(dim):
        single_x_vecs.append(np.linspace(
            x_lim[ii, 0], x_lim[ii, 1], n_for_each_dim))
    x = np.dstack((np.meshgrid(*single_x_vecs))).reshape(-1, dim)
    kern_K = kern.K(x, x) + 1e-5 * np.eye(len(x))

    draw = np.random.multivariate_normal(
        np.zeros(len(x)), kern_K).reshape(-1, 1)

    model = GPy.models.GPRegression(x, draw, kern)
    model.optimize()

    def pred_func(t):
        t = t.reshape(-1, dim)
        m, v = model.predict(t)
        return m

    # __name__ is used to define the filename of an experiment,
    # so making it more descriptive here.
    pred_func.__name__ = "draw-from-gp-{}d".format(dim)

    return pred_func


def egg(x):
    """Eggholder function

    2D function https://www.sfu.ca/~ssurjano/egg.html

    True range is [-512, 512] and true opt is at (512, 404.2319)

    Scaled to [-1, 1], so minimum is now at (1, 0.78951543)

    value at min = -9.596407

    Parameters
    ----------
    x

    Returns
    -------

    """
    x = 512 * np.atleast_2d(x)
    x0 = x[:, 0]
    x1 = x[:, 1]
    term1 = -(x1 + 47) * np.sin(np.sqrt(np.abs(x1 + x0 / 2 + 47)))
    term2 = -x0 * np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
    y = (term1 + term2) / 100
    return y[:, None]


def shekel(x: np.ndarray) -> np.ndarray:
    """Shekel

    Actual domain is [0, 1]

    this is shifted and scaled to [-1, 1]

    Parameters
    ----------
    x

    """
    x = (x + 1) / 2
    x = np.atleast_2d(x)
    m = 10
    b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])[:, None]
    C = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                  [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                  [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                  [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
    outer = 0.0
    for i in range(m):
        bi = b[i]
        inner = 0
        for j in range(4):
            xj = x[:, j] * 10
            Cji = C[j, i]
            inner = inner + (xj - Cji) ** 2

        outer = outer + 1 / (inner + bi)

    y = - outer
    return y


def twosines(x: np.ndarray) -> np.ndarray:
    """Sum of two unequal sinusoids. 1-D problem

    Actual domain is [2.7, 7.5] with min_loc = 5.145735 and f_opt = -1.899599

    this is shifted and scaled to [-1, 1], so min_loc = 0.019056249999999997

    Parameters
    ----------
    x

    """
    x = (x + 1) / 2 * (7.5 - 2.7) + 2.7
    res = np.sin(x) + np.sin(10 / 3 * x)
    res = res.sum(-1)

    return res


def hartmann6(x):
    """
    https://github.com/automl/HPOlib2/blob/master/hpolib/benchmarks
    /synthetic_functions/hartmann6.py

    in original [0, 1]^6 space:
    global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    min function value = -3.32237
    """
    # Re-scale input space from [-1, 1]^6 (X_LIM) to
    # [0,1]^6 (original definition of the function)
    # so that code below still works
    x = (x + 1) / 2.

    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                  [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                  [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                  [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                           [2329, 4135, 8307, 3736, 1004, 9991],
                           [2348, 1451, 3522, 2883, 3047, 6650],
                           [4047, 8828, 8732, 5743, 1091, 381]])

    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(6):
            internal_sum += A[i, j] * (x[:, j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)

    return external_sum


def ackley(x):
    """
    Ackley function
    x has to be 2D (NxD)
    Bounds = -1, 1
    min x = np.zeros
    y = 0
    """
    x = np.atleast_2d(x).copy()
    x *= 32.768  # rescale x to [-32.768, 32.768]
    n = x.shape[1]
    a = 20
    b = 0.2
    c = 2 * np.pi
    ccx = np.cos(c * x)
    return -a * np.exp(-b * np.sqrt(np.sum(x ** 2, 1) / n)) - \
           np.exp(np.sum(ccx, 1) / n) + a + np.exp(1)


def ackley_small(x):
    """
    Ackley function / 20
    """
    return ackley(x) / 20


def branin(x):
    """
    Branin function -- x = Nx2
    limits: -1 < x < 1

    unscaled x or y:
    Min = 0.1239 0.8183
    Min = 0.5428 0.1517  => 0.3979
    Min = 0.9617 0.1650

    for scaled x and y (current implementation)

    Min = -0.7522 0.6366
    Min = 0.08559 -0.6966  => 0.0013
    Min = 0.9234-0.6699

    """
    x = np.atleast_2d(x)
    # scaling x from [-1, 1] to [0, 1] to keep remaining func unchanged
    x = (x + 1) / 2
    a = x[:, 0] * 15 - 5
    b = x[:, 1] * 15

    # dividing output by 300 to reduce the range of the
    # function to roughly [0, 1.03]
    return ((b - (5.1 / (4 * np.pi ** 2)) * a ** 2 + 5 * a / np.pi - 6) ** 2 +
            10 * (1 - 1 / (8 * np.pi)) * np.cos(a) + 10) / 300


def camelback(x):
    """
    Camelback function
    -1 < x[:,0] < 1
    -1 < x[:,1] < 1

    Global minima: f(x) = -1.0316 (0.0449, -0.7126), (-0.0449, 0.7126)
    """
    x = np.atleast_2d(x).copy()
    x[:, 0] = x[:, 0] * 2

    tmp1 = (4 - 2.1 * x[:, 0] ** 2 + (x[:, 0] ** 4) / 3) * x[:, 0] ** 2
    tmp2 = x[:, 0] * x[:, 1]
    tmp3 = (-4 + 4 * x[:, 1] ** 2) * x[:, 1] ** 2
    return tmp1 + tmp2 + tmp3


def camelback_small(x):
    """
    Camelback function / 5
    """
    return camelback(x) / 5


def michalewicz(x):
    """
    Michalewicz Function
    %old Bounds [0,pi]
    % Bounds [-1, 1]
    %Min = -4.687 (n=5)
    min loc (before transforming x) [2.20, 1.57]
    min loc = [0.31885383,  0.]
    """
    x = np.atleast_2d(x)

    x = (x + 1) * np.pi / 2  # transform from [0,pi] to [-1,1]

    n = x.shape[1]
    m = 10
    ii = np.arange(1, n + 1)

    return -np.sum(np.sin(x) * (np.sin(ii * x ** 2 / np.pi)) ** (2 * m), 1)


def quadratic(x):
    """
    Simple quadratic function
    min at x = 0.53
    f(x_min) = 10.2809
    """

    x = np.atleast_2d(x)
    x -= 0.53
    return np.sum(x ** 2, 1) + 10


def rosenbrock(x):
    """
    Rosenbrock function 2D
    -1 < x < 1
    min y = 0
    """
    x = np.atleast_2d(x)
    x /= 2.048  # scaling x from [-2.048, 2.048] to [-1, 1]
    return (1 - x[:, 0]) ** 2 + 100 * (x[:, 1] - x[:, 0] ** 2) ** 2


def rosenbrock_small(x):
    """
    Rosenbrock function / 50
    """
    return rosenbrock(x) / 50


def levy(x):
    """
    Levy function
    https://www.sfu.ca/~ssurjano/levy.html

    f(x_min) = 0

    """
    x = np.atleast_2d(x * 10.)

    def w(xx, ii):
        """Helper function"""
        return 1 + (xx[:, ii] - 1) / 4

    return np.sin(np.pi * w(x, 0)) ** 2 + \
           np.sum(np.hstack([(w(x, ii) - 1) ** 2 * (
                   1 + 10 * np.sin(np.pi * w(x, ii) + 1) ** 2)
                             for ii in range(x.shape[1] - 1)]), axis=0) + \
           (w(x, -1) - 1) ** 2 * (1 + np.sin(2 * np.pi * w(x, -1) + 1) ** 2)



def get_function(target_func, big=False) \
        -> Tuple[Callable, np.ndarray, np.ndarray, float]:
    """
    Get objects and limits for a chosen function

    Returns (f, X_LIM, min_loc, min_val)
    """
    min_val = None

    if target_func.startswith('camelback-2d'):
        min_loc = np.array([[0.0449, -0.7126], [-0.0449, 0.7126]])

        if not big:
            f = camelback_small
        else:
            f = camelback

        min_val = f(min_loc)

        X_LIM = np.array([[-1, 1], [-1, 1]])

    elif target_func.startswith('twosines'):
        dim = get_dim_from_name(target_func)
        if dim is None:
            dim = 1
        f = twosines
        min_loc = np.array([0.019056249999999997] * dim)
        min_val = -1.899599 * dim
        X_LIM = np.vstack([[-1, 1]] * dim)

    elif target_func == 'hartmann-6d':
        f = hartmann6
        min_loc = np.array(
            [-0.59662, -0.699978, -0.046252, -0.449336, -0.376696, 0.3146])
        min_val = -3.32237
        X_LIM = np.vstack([[-1, 1]] * 6)

    elif target_func.startswith('branin-2d'):
        f = branin
        min_loc = np.array([0.08559, -0.6966])
        min_val = 0.0013
        X_LIM = np.array([[-1, 1], [-1, 1]])

    elif target_func.startswith('egg-2d'):
        f = egg
        min_loc = np.array([1, 0.78951543])
        min_val = -9.596407
        X_LIM = np.array([[-1, 1], [-1, 1]])

    elif target_func.startswith('michalewicz'):
        f = michalewicz
        dim = get_dim_from_name(target_func)
        if dim is None:
            dim = 2

        if dim == 2:
            min_loc = np.array([0.3188, 0.])
        elif dim == 3:
            min_loc = np.array([0.3188, 0., -0.1694])
        elif dim == 4:
            min_loc = np.array([0.3191, 0., -0.1691, 0.2203])
        elif dim == 5:
            min_loc = np.array([0.3191, 0., -0.1691, 0.2203, 0.09419])
        else:
            min_loc = None

        if dim == 2:
            min_val = -1.8013
        elif dim == 5:
            min_val = -4.687
        elif dim == 10:
            min_val = -9.66015
        else:
            min_val = None

        X_LIM = np.vstack([[-1, 1]] * dim)

    elif target_func.startswith('ackley'):
        dim = get_dim_from_name(target_func)
        if dim is None:
            dim = 2

        min_loc = np.zeros(dim)
        if not big:
            f = ackley_small
        else:
            f = ackley

        min_val = f(min_loc)
        X_LIM = np.vstack([np.array([-1., 1])] * dim)

    elif target_func.startswith('rosenbrock-2d'):

        if not big:
            f = rosenbrock_small
        else:
            f = rosenbrock

        min_loc = np.array([1., 1.])
        min_val = f(min_loc)

        X_LIM = np.array([[-1, 1], [-1, 1]])

    elif target_func.startswith('levy'):
        f = levy

        dim = get_dim_from_name(target_func)
        if dim is None:
            dim = 2

        min_loc = np.ones(dim)
        min_val = 0
        X_LIM = np.vstack([np.array([-1., 1])] * dim)

    elif target_func == 'shekel-4d':
        f = shekel
        min_loc = np.array([[-0.19985063, -0.20009811,
                             -0.19985063, -0.20009811]])
        min_val = shekel(min_loc)
        X_LIM = np.vstack([[-1, 1]] * 4)

    elif target_func.startswith('quadratic-2d'):
        f = quadratic
        min_loc = np.array([0.53, 0.53])
        min_val = 10.2809
        X_LIM = np.array([[-1, 1], [-1, 1]])


    else:
        print("target_func with name", target_func, "doesn't exist!")
        raise NotImplementedError

    f.__name__ = target_func
    return f, X_LIM, min_loc, min_val


def get_dim_from_name(target_func):
    if len(target_func.split("-")) > 1:
        dim = int(target_func.split("-")[1][:-1])
    else:
        dim = None
    return dim


def plot_test_func(target_func, *args):
    import pylab as plt
    f, X_LIM, _, _ = get_function(target_func, *args)
    n_x1 = 800
    n_x2 = 800
    x1_vals = np.linspace(X_LIM[0, 0], X_LIM[0, 1], n_x1)
    x2_vals = np.linspace(X_LIM[1, 0], X_LIM[1, 1], n_x2)[::-1]

    xx = np.dstack(np.meshgrid(x1_vals, x2_vals)).reshape(-1, 2)

    yy = f(xx)
    yy_grid = yy.reshape(n_x2, n_x1)

    plt.figure(figsize=(7, 7))
    plt.imshow(yy_grid, extent=(
        x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()))
    plt.title(target_func)
    print("function values between {:.2f} and {:.2f}".format(np.min(yy),
                                                             np.max(yy)))
