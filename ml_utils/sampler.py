# -*- coding: utf-8 -*-

import numpy as np


def request_noisy_sample(x, f=None, sigma_x=0, sigma_y=0):
    """
    Returns samples at input locations provided (noise etc is applied here)
    sigma_x and sigma_y are standard deviations

    if f is not provided, then noise is simply applied to the inputs x
    """
    delta_x = np.random.standard_normal(x.shape) * sigma_x
    x_sample = x + delta_x
    if f is not None:
        f_of_x = f(x).reshape(-1, 1)
        f_of_x_sample = f(x_sample).reshape(-1, 1)
        delta_y = np.random.standard_normal(f_of_x_sample.shape) * sigma_y
        f_sample = f_of_x_sample + delta_y
    else:
        f_of_x = None
        f_of_x_sample = None
        delta_y = None
        f_sample = None

    return {'f_sample': f_sample,
            'x_sample': x_sample,
            'f_of_x': f_of_x,
            'f_of_x_sample': f_of_x_sample,
            'delta_x': delta_x,
            'delta_y': delta_y}
