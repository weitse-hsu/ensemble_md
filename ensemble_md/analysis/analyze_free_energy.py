####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################
"""
The :obj:`.analyze_free_energy` module provides functions for performing free energy
calculations for REXEE simulations.
"""
import alchemlyb
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pymbar.timeseries import detect_equilibration, subsample_correlated_data  # noqa: E402
from alchemlyb.estimators import TI, BAR, MBAR  # noqa: E402
from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk  # noqa: E402
from alchemlyb.preprocessing import subsampling  # noqa: E402
from ensemble_md.utils import utils  # noqa: E402
from ensemble_md.utils.exceptions import ParameterError  # noqa: E402


def preprocess_data(files_list, temp, data_type, spacing=1, t=None, g=None):
    """
    This function preprocesses :math:`u_{nk}` or :math:`dH/dλ` data for all replicas in an REXEE simulation.
    For each replica, it reads in :math:`u_{nk}` or :math:`dH/dλ` data from all iterations, concatenate
    them, remove the equilibrium region and and decorrelate the concatenated data.

    Parameters
    ----------
    files_list : list
        A list of lists of naturally sorted DHDL file names from all iterations for different replicas.
        Specifically, :code:`files[i]` should be the list of DHDL file names from all iterations of replica :code:`i`.
    temp : float
        The simulation temperature in Kelvin. We assume all replicas were performed at the same temperature.
    data_type : str
        The type of energy data to be procssed. Should be either :code:`'u_nk'` or :code:`'dhdl'`.
    spacing : int, Optional
        The spacing (in the number of data points) to consider when subsampling the data, which is assumed to
        be the same for all replicas. The default is 1.
    t : int, Optional
        The user-specified index that indicates the start of equilibrated data. If this parameter is not specified,
        the function will estimate it using :func:`pymbar.timeseries.detect_equilibration`. The default
        is :code:`None`.
    g : int, Optional
        The user-specified statistical inefficiency. If this parameter is not specified, the function will estimate
        it using :func:`pymbar.timeseries.detect_equilibration`. The default is :code:`None`.

    Returns
    -------
    preprocessed_data_all : pandas.Dataframe
        A list of preprocessed :math:`u_{nk}` or :math:`dH/dλ` data for all replicas that can serve as the
        input to free energy estimators.
    t_list : list
        A list of indices indicating the start of equilibrated data for different replicas. This list will
        be empty if the parameter :code:`t` is specified.
    g_list : list
        A list of statistical inefficiencies of the equilibrated data for different replicas. This list will
        be empty if the parameter :code:`g` is specified.
    """
    if data_type == 'u_nk':
        extract_fn, convert_fn = extract_u_nk, subsampling.u_nk2series
    elif data_type == 'dhdl':
        extract_fn, convert_fn = extract_dHdl, subsampling.dhdl2series
    else:
        raise ValueError("Invalid data_type. Expected 'u_nk' or 'dhdl'.")

    user_specified = None
    if t is None or g is None:
        user_specified = False

    n_sim = len(files_list)
    preprocessed_data_all, t_list, g_list = [], [], []
    for i in range(n_sim):
        print(f'Reading dhdl files of state range {i} ...')
        print(f'Collecting {data_type} data from all iterations ...')
        data = alchemlyb.concat([extract_fn(xvg, T=temp) for xvg in files_list[i]])
        data_series = convert_fn(data)
        data, data_series = subsampling._prepare_input(data, data_series, drop_duplicates=True, sort=True)
        data = subsampling.slicing(data, step=spacing)
        data_series = subsampling.slicing(data_series, step=spacing)

        if user_specified is False:
            print('Estimating the start index of the equilibrated data and the statistical inefficiency ...')
            t, g, Neff_max = detect_equilibration(data_series.values)
            t_list.append(t)
            g_list.append(g)
        else:
            # we only need to estimate Neff_max here.
            Neff_max = int((len(data_series.values) - t) / g)

        print(f'Subsampling and decorrelating the concatenated {data_type} data ...')
        print(f'  Adopted spacing: {spacing:.0f}')
        print(f'  {t / len(data_series) * 100:.1f}% of the {data_type} data was in the equilibrium region and therfore discarded.')  # noqa: E501
        print(f'  Statistical inefficiency of {data_type}: {g:.1f}')
        print(f'  Number of effective samples: {Neff_max:.0f}\n')

        data_series_equil, data_equil = data_series[t:], data[t:]
        indices = subsample_correlated_data(data_series_equil, g=g)
        preprocessed_data = data_equil.iloc[indices]

        preprocessed_data_all.append(preprocessed_data)

    return preprocessed_data_all, t_list, g_list


def _apply_estimators(data, df_method="MBAR"):
    """
    An internal function used in :func:`calculate_free_energy` to generate a list of estimators fitting the input data.

    Parameters
    ----------
    data : pandas.Dataframe
        A list of :math:`dH/dλ` or :math:`u_{nk}` dataframes obtained from all replicas of the REXEE simulation
        Preferrably, the :math:`dH/dλ` or :math:`u_{nk}` data should be preprocessed by the function
        :func:`preprocess_data`.
    df_method : str, Optional
        The selected free energy estimator. Options include :code:`"MBAR"`, :code:`"BAR"` and :code:`"TI"`.
        The default is :code:`"MBAR"`.

    Returns
    -------
    estimators : list
        A list of estimators fitting the input data for all replicas. With this, the user
        can access all the free energies and their associated uncertainties for all states and replicas.

    See also
    --------
    :func:`calculate_free_energy`
    """
    n_sim = len(data)
    estimators = []  # A list of objects of the corresponding class in alchemlyb.estimators
    for i in range(n_sim):
        if df_method == "TI":
            estimators.append(TI().fit(data[i]))
        elif df_method == "BAR":
            estimators.append(BAR().fit(data[i]))
        elif df_method == "MBAR":
            estimators.append(MBAR().fit(data[i]))
        else:
            raise ParameterError('Specified estimator not available.')

    return estimators


def _calculate_df_adjacent(estimators):
    """
    An internal function used in :func:`calculate_free_energy` to calculate a list of free energies between adjacent
    states for all replicas.

    Parameters
    ----------
    estimators : list
        A list of estimators fitting the input data for all replicas. With this, the user
        can access all the free energies and their associated uncertainties for all states and replicas.
        In our code, these estimators come from the function :func:`_apply_estimators`.

    Returns
    -------
    df_adjacent : list
        A list of lists free energy differences between adjacent states for all replicas.
    df_err_adjacent : list
        A list of lists of uncertainties corresponding to the values in :code:`df_adjacent`.

    See also
    --------
    :func:`calculate_free_energy`
    """
    n_sim = len(estimators)
    df_adjacent = [list(np.array(estimators[i].delta_f_)[:-1, 1:].diagonal()) for i in range(n_sim)]
    df_err_adjacent = [list(np.array(estimators[i].d_delta_f_)[:-1, 1:].diagonal()) for i in range(n_sim)]

    return df_adjacent, df_err_adjacent


def _combine_df_adjacent(df_adjacent, state_ranges, df_err_adjacent=None, err_type="propagate"):
    """
    An internal function used in :func:`calculate_free_energy` to combine the free energy differences between
    adjacent states in different state ranges using either simple means or inverse-variance weighted means.

    Parameters
    ----------
    df_adjacent : list
        A list of lists free energy differences between adjacent states for all replicas.
    state_ranges : list
        A list of lists of showing the state indices sampled by each replica.
    n_tot : int
        Number of lambda states
    df_err_adjacent : list, Optional
        A list of lists of uncertainties corresponding to the values of :code:`df_adjacent`. Notably, if
        :code:`df_err_adjacent` is :code:`None`, simple means will be used. Otherwise, inverse-variance weighted
        means will be used. The default is :code:`None`
    err_type : str, Optional
        How the error of the combined free energy differences should be calculated. Available options include
        :code:`"propagate"` and :code:`"std"`. Note that the option :code:`"propagate"` is only available when
        :code:`df_err_adjacent` is not :code:`None`.

    Returns
    -------
    df : list
        A list of free energy differences between states :math:`i` and :math:`i + 1` for the entire state range.
    df_err : list
        A list of uncertainties of the free energy differences for the entire state range.
    overlap_bool : list
        A list of boolean values indicating whether a free energy difference was available in multiple replicas.
        For example, :code:`overlap_bool[i] = True` means that the :math:`i`-th free energy difference (i.e.
        :code:`df[i]`) was available in multiple replicas.

    See also
    --------
    :func:`calculate_free_energy`
    """
    n_tot = state_ranges[-1][-1] + 1
    df, df_err, overlap_bool = [], [], []
    for i in range(n_tot - 1):
        # df_list is a list of free energy difference between sates i and i+1 in different replicas
        # df_err_list contains the uncertainties corresponding to the values of df_list
        df_list, df_err_list = [], []
        for j in range(len(state_ranges)):   # len(state_ranges) = n_sim
            if i in state_ranges[j] and i + 1 in state_ranges[j]:
                idx = state_ranges[j].index(i)
                df_list.append(df_adjacent[j][idx])
                if df_err_adjacent is not None:
                    df_err_list.append(df_err_adjacent[j][idx])
        overlap_bool.append(len(df_list) > 1)

        if df_err_adjacent is None:
            # simple means and std will be used
            mean, error = np.mean(df_list), np.std(df_list, ddof=1)
        else:
            # inverse-variance weighted means and propagated error will be used
            mean, error = utils.weighted_mean(df_list, df_err_list)

            if err_type == 'std':
                if len(df_list) == 1:
                    error = df_err_list[0]
                else:
                    # overwrite the error calculated above
                    error = np.std(df_list, ddof=1)

        df.append(mean)
        df_err.append(error)

    return df, df_err, overlap_bool


def calculate_free_energy(data, state_ranges, df_method="MBAR", err_method="propagate", n_bootstrap=None, seed=None, MTREXEE=False):  # noqa: E501
    """
    Caculates the averaged free energy profile with the chosen method given :math:`u_{nk}` or :math:`dH/dλ` data
    obtained from all replicas of the REXEE simulation. Available methods include TI, BAR, and MBAR. TI
    requires :math:`dH/dλ` data while the other two require :math:`u_{nk}` data.

    Parameters
    ----------
    data : pandas.Dataframe
        A list of :math:`u_{nk}` or :math:`dH/dλ` dataframes obtained from all replicas of the REXEE simulation.
        Preferrably, the :math:`u_{nk}` or :math:`dH/dλ` data should be preprocessed by the function
        :func:`proprocess_data`.
    state_ranges : list
        A list of lists of showing the state indices sampled by each replica.
    df_method : str, Optional
        The method used to calculate the free energy profile. Available choices include :code:`"TI"`,
        :code:`"BAR"`, and :code:`"MBAR"`. The default is :code:`"MBAR"`.
    err_method : str, Optional
        The method used to estimate the uncertainty of the free energy combined across multiple replicas.
        Available options include :code:`"propagate"` and :code:`"bootstrap"`. The bootstrapping method is
        more accurate but much more computationally expensive than simple error propagation.
    n_bootstrap : int, Optional
        The number of bootstrap iterations. This parameter is used only when the boostrapping method is chosen to
        estimate the uncertainties of the free energies. The default is :code:`None`. In the CLI :code:`analyze_REXEE`,
        this number is set by the YAML parameter :code:`n_bootstrap`.
    seed : int, Optional
        The random seed for bootstrapping. Only relevant when :code:`err_method` is :code:`"bootstrap"`.
        The default is :code:`None`.
    MTREXEE : bool
        Whether this is a MT-REXEE simulation or not

    Returns
    -------
    f : list
        The full-range free energy profile.
    f_err : list
        The uncertainties corresponding to the values in :code:`f`.
    estimators : list
        A list of estimators fitting the input data for all replicas. With this, the user
        can access all the free energies and their associated uncertainties for all states and replicas.

    Example
    -------
    In the CLI :code:`analyze_REXEE`, lines like below are used:

        >>> import glob
        >>> import natsort
        >>> from ensemble_md.analysis import analyze_free_energy
        >>> state_ranges = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
        >>> file_list = [natsort.natsorted(glob.glob(f'sim_{i}/iteration_*/dhdl*xvg')) for i in range(4)]
        >>> data_list, _, _ = analyze_free_energy.preprocess_data(file_list, temp=300, data_type='u_nk')
        >>> f, _, _ = analyze_free_energy.calculate_free_energy(data_list, state_ranges, "MBAR", "propagate")
    """
    n_sim = len(data)
    if MTREXEE is False:
        n_tot = state_ranges[-1][-1] + 1
    else:
        n_tot = state_ranges[-1] + 1
    estimators = _apply_estimators(data, df_method)
    if MTREXEE is False:
        df_adjacent, df_err_adjacent = _calculate_df_adjacent(estimators)
        df, df_err, overlap_bool = _combine_df_adjacent(df_adjacent, state_ranges, df_err_adjacent, err_type='propagate')  # noqa: E501
    else:
        df, df_err = _calculate_df_adjacent(estimators)

    if err_method == 'bootstrap':
        if seed is not None:
            print(f'Setting the random seed for boostrapping: {seed}')

        # Recalculate err with bootstrapping. (df is still the same and has been calculated above.)
        df_bootstrap = []
        sampled_data_all = [data[i].sample(n=len(data[i]) * n_bootstrap, replace=True, random_state=seed) for i in range(n_sim)]  # noqa: E501
        for b in range(n_bootstrap):
            sampled_data = [sampled_data_all[i].iloc[b * len(data[i]):(b + 1) * len(data[i])] for i in range(n_sim)]
            bootstrap_estimators = _apply_estimators(sampled_data, df_method)
            if MTREXEE is False:
                df_adjacent, df_err_adjacent = _calculate_df_adjacent(bootstrap_estimators)
                df_sampled, _, overlap_bool = _combine_df_adjacent(df_adjacent, state_ranges, df_err_adjacent, err_type='propagate')  # doesn't matter what value err_type here is # noqa: E501
            else:
                df_sampled, _ = _calculate_df_adjacent(bootstrap_estimators)
            df_bootstrap.append(df_sampled)
        error_bootstrap = np.std(df_bootstrap, axis=0, ddof=1)

        # Replace the value in df_err with value in error_bootstrap if df_err corresponds to
        # the df between overlapping states
        for i in range(n_tot - 1):
            if MTREXEE is True or overlap_bool[i] is True:
                print(f'Replaced the propagated error with the bootstrapped error for states {i} and {i + 1}: {df_err[i]:.5f} -> {error_bootstrap[i]:.5f}.')  # noqa: E501
                df_err[i] = error_bootstrap[i]
    elif err_method == 'propagate':
        pass
    else:
        raise ParameterError('Specified err_method not available.')

    df.insert(0, 0)
    df_err.insert(0, 0)
    f = [sum(df[:(i + 1)]) for i in range(len(df))]
    f_err = [np.sqrt(sum([x**2 for x in df_err[:(i+1)]])) for i in range(len(df_err))]

    return f, f_err, estimators


def calculate_df_rmse(estimators, df_ref, state_ranges):
    """
    Calculates the RMSE values of the free energy profiles of different state ranges given the reference free
    energy profile for the whole range of states.

    Parameters
    ----------
    estimators : list
        A list of estimators fitting the input data for all replicas. With this, the user
        can access all the free energies and their associated uncertainties for all states and replicas.
        The estimators should be generated by the function :func:`calculate_free_energy`.
    df_ref : list
        A list of values corresponding to the free energies of the whole range of states. The length
        of the list should be equal to the number of states in total.
    state_ranges : list
        A list of lists of showing the state indices sampled by each replica.

    Returns
    -------
    rmse_list : list
        A list of RMSE values of the free energy profiles of different state ranges.

    See also
    --------
    :func:`calculate_free_energy`
    """
    n_sim = len(estimators)
    df_ref = np.array(df_ref)
    rmse_list = []
    for i in range(n_sim):
        df = np.array(estimators[i].delta_f_.iloc[0])  # the first state always has 0 free energy here
        ref = df_ref[state_ranges[i]]
        ref -= ref[0]   # shift the free energy of the first state in the range to 0
        print(df)
        print(ref)
        rmse_list.append(np.sqrt(np.sum((df - ref) ** 2) / len(df)))

    return rmse_list


def plot_free_energy(f, f_err, fig_name):
    """
    Plots the free energy profile with error bars.

    Parameters
    ----------
    f : list
        The full-range free energy profile.
    f_err : list
        The uncertainties corresponding to the values in :code:`f`.
    fig_name : str
        The file path to save the figure.
    """
    plt.figure()
    plt.plot(range(len(f)), f, 'o-', c='#1f77b4')
    plt.errorbar(range(len(f)), f, yerr=f_err, fmt='o', capsize=2, c='#1f77b4')
    plt.xlabel('State')
    plt.ylabel('Free energy (kT)')
    plt.grid()
    plt.savefig(f'{fig_name}', dpi=600)


def average_weights(g_vecs, frac):
    """
    Given the time series of the whole range of alchemical weights, averages the
    weight differences between the the coupled and decoupled states. This can be
    an estimate of the free energy difference between two end states. This function
    is only relevant for weight-updating REXEE simulations.

    Parameters
    ----------
    g_vecs : numpy.ndarray
        An array of alchemical weights of the whole range of states as a function of
        simulation time, which is typically generated by :obj:`.combine_weights`.
    frac : float
        The fraction of :code:`g_vecs` to average over. :code:`frac=0.2` means average
        the last 20% of the weight vectors will be averaged.

    Returns
    -------
    dg_avg : float
        The averaged weight difference between the coupled and decoupled states.
    dg_avg_err : float
        The errors corresponding to the value of :code:`dg_avg`.
    """
    N = len(g_vecs)
    dg = []
    for i in range(N):
        dg.append(g_vecs[i][-1] - g_vecs[i][0])
    n = int(np.floor(N * frac))
    if n <= 1:
        print('The number of samples to be averaged is less than 2, so all samples will be averaged.')
    dg_avg = np.mean(dg[-n:])
    dg_avg_err = np.std(dg[-n:], ddof=1)

    return dg_avg, dg_avg_err
