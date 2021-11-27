import numpy as np
import copy
from slice_sampling import slice_sample


class ModelCollection(object):
    """
    Wrapper class for sampling-based inference.

    This class will hold the sampled hyperparameters and a
    model instance for each sample. It will take care of predictions,
    setting data, etc by forwarding the function to each model instance.

    This should speed up the prediction step, as there isn't an update()
    run for each sample before a prediction is returned.
    """

    def __init__(self, model, verbose=False):
        # make sure we have a sampling-based model
        assert hasattr(model, 'opt_params')
        assert model.opt_params['method'] in ['slice']

        self.verbose = verbose

        self.opt_params = model.opt_params

        # How many samples to discard before keeping slice samples
        if 'burn_in' in self.opt_params.keys():
            self.burn_in = self.opt_params['burn_in']
        else:
            self.burn_in = 100

        model.mode = 'normal'
        model.opt_params = {'method': 'grad'}

        self.original_model = copy.deepcopy(model)
        self.n_samples = self.opt_params['n_samples']
        # self.samples = self.original_model.opt_params['slice_hps']
        self.samples = None

        # create the models for the hp samples
        self.model_list = []
        for ii in range(self.opt_params['n_samples']):
            self.model_list.append(copy.deepcopy(model))

    @property
    def param_array(self):
        """
        The slice sampling hps. This function is defined in order to
        have a unified interface between ModelCollection and standard
        GP models.
        """
        return self.samples

    @param_array.setter
    def param_array(self, p):
        """
        Setting param_array updates the hps in all the local models
        """
        self.samples = p
        self.opt_params['slice_hps'] = p
        self.update_local_models_hps()

    def __getattr__(self, name):
        """
        This redirects any methods that are not explicitly set
        to self.original_model.

        This function is only called if the called attribute
        doesn't already exist.

        If original_model.name is a method, then it will be run.
        Otherwise the element will just be accessed.
        """
        if self.verbose:
            print("Calling function '{}' of embedded model".format(name))
        if callable(getattr(self.original_model, name)):
            try:
                return getattr(self.original_model, name)(*args, **kwargs)
            except NameError:  # no args provided
                return getattr(self.original_model, name)()
        else:
            return getattr(self.original_model, name)

    def plot(self, *args, **kwargs):
        # Use the plotting function in basic_gp but provide the current class
        # for computing the posterior etc.
        # Not pretty, but gets the job done.
        return self.original_model.plot(self, *args, **kwargs)

    def optimize(self, *args, verbose=False, **kwargs):
        """
        Run original_model.optimize() and then update model_list
        """
        # Sample the hps here and store them
        starting_hps = self.original_model.param_array

        if 'hp_bounds' in self.opt_params.keys():
            bounds = np.log(self.opt_params['hp_bounds'])
        else:
            bounds = None
        n_samples = self.opt_params['n_samples']

        if 'sigma' in self.opt_params.keys():
            sigma = self.opt_params['sigma']
        else:
            sigma = 1.0  # the function's default value

        slice_hps = np.zeros((n_samples, len(self.original_model.param_array)))

        for ii in range(self.burn_in):
            if verbose:
                print("Burning in... {}/{}...".format(ii, self.burn_in - 1))

            slice_sample(np.log(self.original_model.param_array),
                         lambda x: -1 *
                         self.original_model.objective_log_theta(x),
                         sigma=sigma,
                         bounds=bounds,
                         verbose=verbose)
        for ii in range(n_samples):
            if verbose:
                print("Getting {}-th sample...".format(ii))
            slice_hps[ii, :] = np.exp(
                slice_sample(np.log(self.original_model.param_array),
                             lambda x: -1 *
                             self.original_model.objective_log_theta(x),
                             sigma=sigma,
                             bounds=bounds,
                             verbose=verbose)
            )
            # self.param_array = starting_hps

        # Set the hps back to the starting ones instead of the last sample
        # of the slice sampling procedure -- why?
        # self.original_model.param_array = starting_hps
        # self.opt_params['slice_hps'] = slice_hps
        self.param_array = slice_hps

        # self.update_local_models_hps()

    def update_local_models_hps(self):
        """
        Sets the hyperparameters of model list
        """
        for ii in range(len(self.model_list)):
            self.model_list[ii].param_array = self.samples[ii]

    def update_local_models_data(self):
        """
        Sets the X and Y variables in model list
        """
        for ii in range(len(self.model_list)):
            self.model_list[ii].set_XY(X=self.original_model.X,
                                       Y=(self.original_model.Y +
                                          self.original_model.y_mean))

    def set_XY(self, *args, **kwargs):
        """
        GPy models have this interface and I'm using this interface in parts
        of my code
        """
        self.set_data(*args, **kwargs)

    def set_data(self, *args, **kwargs):
        """
        All models have this function, but I'm not sure if I'm using
        this interface. Keeping it for now.
        """
        self.original_model.set_data(*args, **kwargs)
        self.update_local_models_data()

    def predict(self, x_star, y_star=None, full_cov=False):
        # Predict using each sampled hp and return the mean posterior
        mu = np.zeros((len(x_star), 1))
        var = np.zeros((len(x_star), 1))

        # no need to hold on to each sample's values, so just adding
        # them all together and then dividing by n after the loop
        for ii in range(self.n_samples):
            m, v = self.model_list[ii].predict(x_star, full_cov=full_cov)
            mu += m
            var += v
        mu = mu / len(self.samples)
        var = var / len(self.samples)

        if y_star is not None:
            if full_cov:
                _var = np.diag(var)
            else:
                _var = var
            log_prob = -0.5 * (np.log(2 * np.pi) + np.log(_var) +
                               (y_star - mu)**2 / _var)
            return mu, var, log_prob
        return mu, var
