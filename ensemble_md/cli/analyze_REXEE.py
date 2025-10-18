####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################
from ensemble_md.utils import utils  # noqa: E402
from ensemble_md.utils import gmx_parser  # noqa: E402
from ensemble_md.analysis import analyze_traj  # noqa: E402
from ensemble_md.analysis import analyze_matrix  # noqa: E402
from ensemble_md.analysis import msm_analysis  # noqa: E402
from ensemble_md.analysis import analyze_free_energy  # noqa: E402
from ensemble_md.replica_exchange_EE import ReplicaExchangeEE  # noqa: E402
from ensemble_md.utils.exceptions import ParameterError  # noqa: E402
import os
import sys
import time
import glob
import pyemma
import pymbar
import pickle
import natsort
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from datetime import datetime
from deeptime.markov.tools.analysis import is_transition_matrix
import pandas as pd
warnings.simplefilter(action='ignore', category=UserWarning)


def initialize(args):
    parser = argparse.ArgumentParser(
        description='This CLI analyzes a REXEE simulation.')
    parser.add_argument('-y',
                        '--yaml',
                        type=str,
                        default='params.yaml',
                        help='The file path of the input YAML file used to run the REXEE simulation. \
                            (Default: params.yaml)')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='analyze_REXEE_log.txt',
                        help='The file path of the output log file that contains the analysis results of REXEE. \
                            (Default: analyze_REXEE_log.txt)')
    parser.add_argument('-rt',
                        '--rep_trajs',
                        type=str,
                        default='rep_trajs.npy',
                        help='The file path of the NPY file containing the replica-space trajectory. \
                            (Default: rep_trajs.npy)')
    parser.add_argument('-st',
                        '--state_trajs',
                        type=str,
                        default='state_trajs.npy',
                        help='The file path of the NPY file containing the stitched state-space trajectory. \
                            If the specified file is not found, the code will try to find all the trajectories and \
                            stitch them. (Default: state_trajs.npy)')
    parser.add_argument('-sts',
                        '--state_trajs_for_sim',
                        type=str,
                        default='state_trajs_for_sim.npy',
                        help='The file path of the NPY file containing the stitched state-space time series for \
                             different state sets. If the specified file is not found, the code will try to find \
                             all the time series and stitch them. (Default: state_trajs.npy)')
    parser.add_argument('-d',
                        '--dir',
                        default='analysis',
                        help='The path of the folder for storing the analysis results. (Default: analysis)')
    parser.add_argument('-m',
                        '--maxwarn',
                        type=int,
                        default=0,
                        help='The maximum number of warnings in parameter specification to be ignored. (Default: 0)')
    args_parse = parser.parse_args(args)

    return args_parse


def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    sys.stdout = utils.Logger(logfile=args.output)
    sys.stderr = utils.Logger(logfile=args.output)
    section_idx = 0

    rc('font', **{
        'family': 'sans-serif',
        'sans-serif': ['DejaVu Sans'],
    })
    # Set the font used for MathJax - more on this later
    rc('mathtext', **{'default': 'regular'})
    plt.rc('font', family='serif')

    print(f'Current time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    print(f'Command line: {" ".join(sys.argv)}')

    REXEE = ReplicaExchangeEE(args.yaml)
    REXEE.print_params(params_analysis=True)

    for i in REXEE.warnings:
        print()
        print(f'{i}')
        print()

    if len(REXEE.warnings) > args.maxwarn:
        raise ParameterError(
            f"The execution failed due to warning(s) about parameter spcificaiton. Consider setting maxwarn in the input YAML file if you want to ignore them.")  # noqa: E501, F541

    # Check if the folder for saving the outputs exist. If not, create the folder.
    if os.path.exists(args.dir) is False:
        os.mkdir(args.dir)

    print('\nData analysis of the simulation ensemble')
    print('========================================')

    # Section 1. Analysis based on transitions between state sets
    print('[ Section 1. Analysis based on transitions between state sets/replicas ]')
    section_idx += 1

    # 1-0. Read in replica-space trajectories
    print('1-0. Reading in the replica-space trajectory ...')
    rep_trajs = np.load(args.rep_trajs)  # Shape: (n_sim, n_iter)

    # 1-1. Plot the replica-sapce trajectory
    print('1-1. Plotting transitions between state sets/replicas ...')
    dt_swap = REXEE.nst_sim * REXEE.dt    # dt for swapping replicas
    analyze_traj.plot_rep_trajs(rep_trajs, f'{args.dir}/rep_trajs.png', dt_swap)

    # 1-2. Plot the replica-space transition matrix
    print('1-2. Plotting the replica-space transition matrix (considering all continuous trajectories) ...')
    counts = [analyze_traj.traj2transmtx(rep_trajs[i], REXEE.n_sim, normalize=False) for i in range(len(rep_trajs))]
    reps_mtx = np.sum(counts, axis=0)  # First sum up the counts. This should be symmetric if n_ex=1. Otherwise it might not be. # noqa: E501
    reps_mtx /= np.sum(reps_mtx, axis=1)[:, None]   # and then normalize each row
    np.save(f'{args.dir}/reps_transmtx_allconfigs.npy', reps_mtx)
    analyze_matrix.plot_matrix(reps_mtx, f'{args.dir}/rep_transmtx_allconfigs.png')

    # 1-3. Calculate the spectral gap for the replica-space transition amtrix
    if REXEE.modify_coords is None:  # Causes issues for non-continuous MT-REXEE trajectories
        print('1-3. Calculating the spectral gap of the replica-space transition matrix ...')
        spectral_gap, spectral_gap_err, eig_vals = analyze_matrix.calc_spectral_gap(reps_mtx)
        print(f'The spectral gap of the replica-space transition matrix: {spectral_gap:.3f}')

    # Section 2. Analysis based on transitions between states
    print('\n[ Section 2. Analysis based on transitions between states ]')
    section_idx += 1

    # 2-1. Recover all continuous trajectories
    if os.path.isfile(args.state_trajs) is True:
        print('2-1. Reading in the stitched state-space trajectory ...')
        state_trajs = np.load(args.state_trajs)
    else:
        # This may take a while.
        print('2-1. Stitching trajectories for each starting configuration from dhdl files ...')
        dhdl_files = [natsort.natsorted(glob.glob(f'sim_{i}/iteration_*/*dhdl*xvg')) for i in range(REXEE.n_sim)]
        shifts = np.arange(REXEE.n_sim) * REXEE.s
        state_trajs = analyze_traj.stitch_time_series(dhdl_files, rep_trajs, shifts=shifts, save_npy=True)  # length: the number of replicas  # noqa: E501

    # 2-2. Plot the state-space trajectory
    print('\n2-2. Plotting transitions between different alchemical states ...')
    dt_traj = REXEE.dt * REXEE.template['nstdhdl']  # in ps
    analyze_traj.plot_state_trajs(
        state_trajs,
        REXEE.state_ranges,
        f'{args.dir}/state_trajs.png',
        dt_traj
    )

    # 2-3. Plot the histograms of state index for different trajectories
    print('\n2-3. Plotting the histograms of the state index for different trajectories ...')
    hist_data = analyze_traj.plot_state_hist(
        state_trajs,
        REXEE.state_ranges,
        f'{args.dir}/state_hist.png'
    )
    rmse = analyze_traj.calc_hist_rmse(hist_data, REXEE.state_ranges)
    print(f'The RMSE of accumulated histogram counts of the state index: {rmse:.0f}')

    if REXEE.proposal != 'random_range':  # Need to FIX THIS FOR RANDOM-Range
        # 2-4. Stitch the time series of state index for different replicas
        if os.path.isfile(args.state_trajs_for_sim) is True:
            print('\n2-4. Reading in the stitched time series of state index for different replicas ...')
            state_trajs_for_sim = np.load(args.state_trajs_for_sim)
        else:
            # This may take a while.
            print('2-4. Stitching time series of state index for each replica ...')
            shifts = list(REXEE.s * np.arange(REXEE.n_sim))
            dhdl_files = [natsort.natsorted(glob.glob(f'sim_{i}/iteration_*/*dhdl*xvg')) for i in range(REXEE.n_sim)]
            state_trajs_for_sim = analyze_traj.stitch_time_series_for_sim(dhdl_files, shifts)

        # 2-5. Plot the time series of state index for different state sets
        print('\n2-5. Plotting the time series of state index for different state sets ...')
        analyze_traj.plot_state_trajs(
            state_trajs_for_sim,
            REXEE.state_ranges,
            f'{args.dir}/state_trajs_for_sim.png',
            title_prefix='State set'
        )

        # 2-6. Plot the histograms of state index for different state sets
        print('\n2-6. Plotting the histograms of state index for different state sets')
        analyze_traj.plot_state_hist(
            state_trajs_for_sim,
            REXEE.state_ranges,
            f'{args.dir}/state_hist_for_sim.png',
            prefix='State set',
            subplots=True
        )

    # 2-7. Plot the overall state transition matrices calculated from the state-space trajectories
    print('\n2-7. Plotting the overall state transition matrices from different trajectories ...')
    mtx_list = []
    for i in range(REXEE.n_sim):
        mtx = analyze_traj.traj2transmtx(state_trajs[i], REXEE.n_tot)
        mtx_list.append(mtx)
        analyze_matrix.plot_matrix(mtx, f'{args.dir}/traj_{i}_state_transmtx.png')

    # 2-8. For each trajectory, calculate the spectral gap of the overall transition matrix obtained in step 2-2.
    print('\n2-8. Calculating the spectral gap of the state transition matrices ...')
    results = [analyze_matrix.calc_spectral_gap(mtx) for mtx in mtx_list]  # a list of tuples
    spectral_gaps = [results[i][0] if None not in results else None for i in range(len(results))]
    eig_vals = [results[i][2] if None not in results else None for i in range(len(results))]
    if None not in spectral_gaps:
        for i in range(REXEE.n_sim):
            print(f'   - Trajectory {i}: {spectral_gaps[i]:.3f} (λ_1: {eig_vals[i][0]:.5f}, λ_2: {eig_vals[i][1]:.5f})')  # noqa: E501
        print(f'   - Average of the above: {np.mean(spectral_gaps):.3f} (std: {np.std(spectral_gaps, ddof=1):.3f})')

    # 2-9. For each trajectory, calculate the stationary distribution from the overall transition matrix obtained in step 2-2.  # noqa: E501
    if REXEE.modify_coords is None:  # Causes issues for non-continuous MT-REXEE trajectories
        print('\n2-9. Calculating the stationary distributions ...')
        pi_list = [analyze_matrix.calc_equil_prob(mtx) for mtx in mtx_list]
        if any([x is None for x in pi_list]):
            pass  # None is in the list
        else:
            for i in range(REXEE.n_sim):
                print(f'   - Trajectory {i}: {", ".join([f"{j:.3f}" for j in pi_list[i].reshape(-1)])}')
            if len({len(i) for i in pi_list}) == 1:  # all lists in pi_list have the same length
                print(f'   - Average of the above: {", ".join([f"{i:.3f}" for i in np.mean(pi_list, axis=0).reshape(-1)])}')  # noqa: E501

    # 2-10. Calculate the state index correlation time for each trajectory (this step is more time-consuming one)
    if REXEE.modify_coords is None:  # Causes issues for non-continuous MT-REXEE trajectories
        print('\n2-10. Calculating the state index correlation time ...')
        tau_list = [(pymbar.timeseries.statistical_inefficiency(state_trajs[i], fast=True) - 1) / 2 * dt_traj for i in range(REXEE.n_sim)]  # noqa: E501
        for i in range(REXEE.n_sim):
            print(f'   - Trajectory {i}: {tau_list[i]:.1f} ps')
        print(f'   - Average of the above: {np.mean(tau_list):.1f} ps (std: {np.std(tau_list, ddof=1):.1f} ps)')

    # 2-11. Calculate transit times for each trajectory
    print('\n2-11. Plotting the average transit times ...')
    t_0k_list, t_k0_list, t_roundtrip_list, units = analyze_traj.plot_transit_time(state_trajs, REXEE.n_tot, dt=dt_traj, folder=args.dir)  # noqa: E501
    meta_list = [t_0k_list, t_k0_list, t_roundtrip_list]
    t_names = [
        '\n   - Average transit time from states 0 to k',
        '\n   - Average transit time from states k to 0',
        '\n   - Average round-trip time',
    ]
    for i in range(len(meta_list)):
        t_list = meta_list[i]
        print(t_names[i])
        for j in range(len(t_list)):
            if t_list[j] is None:
                print(f'     - Trajectory {j}: Skipped')
            else:
                print(f'     - Trajectory {j} ({len(t_list[j])} events): {np.mean(t_list[j]):.2f} {units}')
        print(f'     - Average of the above: {np.mean([np.mean(i) for i in t_list]):.2f} {units} (std: {np.std([np.mean(i) for i in t_list], ddof=1):.2f} {units})')  # noqa: E501

    if REXEE.msm is True:
        section_idx += 1

        # Section 3. Analysis based on Markov state models
        print('\n[ Section 3. Analysis based on Markov state models ]')

        # 3-1. Plot the implied timescale as a function of lag time
        print('\n3-1. Plotting the implied timescale as a function of lag time for all trajectories ...')
        lags = np.arange(REXEE.lag_spacing, REXEE.lag_max + REXEE.lag_spacing, REXEE.lag_spacing)
        # lags could also be None and decided automatically. Could consider using that.
        ts_list = msm_analysis.plot_its(state_trajs, lags, fig_name=f'{args.dir}/implied_timescales.png', dt=dt_traj, units='ps')  # noqa: E501

        # 3-2. Decide a lag time for building MSM
        print('\n3-2. Calculating recommended lag times for building MSMs ...')
        chosen_lags = msm_analysis.decide_lagtimes(ts_list)
        print('     \nSuggested lag times for each trajectory:')
        for i in range(len(chosen_lags)):
            print(f'       - Trajectory {i}: {chosen_lags[i] * dt_traj:.1f} ps')

        # 3-3. Build a Bayesian MSM and perform a CK test for each trajectory to validate the models
        print('\n3-3. Building Bayesian MSMs for the state-space trajectory for each trajectory ...')
        print('     Performing a Chapman-Kolmogorov test on each trajectory ...')
        models = [pyemma.msm.bayesian_markov_model(state_trajs[i], chosen_lags[i], dt_traj=f'{dt_traj} ps', show_progress=False) for i in range(REXEE.n_sim)]  # noqa: E501

        for i in range(len(models)):
            print(f'     Plotting the CK-test results for trajectory {i} ...')
            mlags = 5  # this maps to 5 (mlags: multiples of lag times for testing the model)
            nsets = models[i].nstates  # number of metastable states.
            # Note that nstates is the number of unique states in the input trajectores counted with the effective mode
            # (see the documentation) Therefore, if a system barely sampled some of the states, those states will
            # not be counted as involved in the transition matrix (i.e. not in the active set). To check the
            # active states, use models[i].active_set. If the system sampled all states frequently,
            # models[i].active_set should be equal to np.unique(state_trajs[i]) and both lengths should be
            # REXEE.n_tot. I'm not sure why the attribute nstates_full is not always REXEE.n_tot but is less
            # relevant here.
            cktest = models[i].cktest(nsets=nsets, mlags=mlags, show_progress=False)
            pyemma.plots.plot_cktest(cktest, dt=dt_traj, units='ps')
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.savefig(f'{args.dir}/traj_{i}_CKtest.png', dpi=600)

        # Additionally, check if the sampling is poor for each trajectory
        for i in range(len(models)):
            if models[i].nstates != REXEE.n_tot:
                print(f'     Note: The sampling of trajectory {i} was poor.')

        # 3-4. Plot the state transition matrices estimated with the specified lag times in MSMs
        print('\n3-4. Plotting the overall transition matrices for all trajectories and their average ...')
        mtx_list = [m.transition_matrix for m in models]
        mtx_list_modified = []  # just for plotting (if all trajs sampled the fulle range frequently, this will be the same as mtx_list)  # noqa: E501
        for i in range(len(mtx_list)):
            # check if each mtx in mtx_list spans the full state set. (If the system did not visit
            # certain states, the dimension will be less than REXEE.n_tot * REXEE.n_tot. In this case, we
            # add rows and columns of 0. Note that the modified matrix will not be a transition matrix,
            # so this is only for plotting. For later analysis such as spectral gap calculation, we
            # will just use the unmodified matrices.
            if mtx_list[i].shape != (REXEE.n_tot, REXEE.n_tot):  # add rows and columns of 0
                sampled = models[i].active_set
                missing = list(set(range(REXEE.n_tot)) - set(sampled))  # states not visited

                # figure out which end we should stack rows/columns to
                n_1 = sum(missing > max(sampled))   # add rows/columns to the end of large state indices
                n_2 = sum(missing < min(sampled))   # add rows/columns to the end of small state indices

                # Below we first add a column and a row and so on
                mtx = mtx_list[i]
                for j in range(n_1):
                    d = mtx.shape[0]  # size of the square matrix
                    mtx = np.hstack((mtx, np.zeros([d, 1])))
                    mtx = np.vstack((mtx, np.zeros([1, d + 1])))
                for j in range(n_2):
                    d = mtx.shape[0]  # size of the square matrix
                    mtx = np.hstack((np.zeros([d, 1]), mtx))
                    mtx = np.vstack((np.zeros([1, d + 1]), mtx))
                mtx_list_modified.append(mtx)
            else:
                mtx_list_modified.append(mtx_list[i])

        avg_mtx = np.mean(mtx_list_modified, axis=0)
        print('     Saving transmtx.npy (plotted transition matrices)...')
        np.save(f'{args.dir}/transmtx.npy', mtx_list_modified)

        for i in range(REXEE.n_sim):
            analyze_matrix.plot_matrix(mtx_list[i], f'{args.dir}/traj_{i}_state_transmtx_msm.png')
        analyze_matrix.plot_matrix(avg_mtx, f'{args.dir}/state_transmtx_avg_msm.png')

        # Note that if a trajectory never visited a certain replica, the rows in the state set of that
        # replica will only have 0 entries. Below we check if this is the case.
        print('     Checking the sum of each row of each transition matrix is 1 ...')
        is_transmtx = [is_transition_matrix(i) for i in mtx_list]
        for i in range(len(is_transmtx)):
            if is_transmtx[i] is False:
                print(f'     The sums of the rows are not 1 (but {np.sum(mtx_list[i], axis=1)}) for the transition matrix of trajectory {i}.')  # noqa: E501

        # 3-5. Calculate the spectral gap from the transition matrix of each trajectory
        print('\n3-5. Calculating the spectral gap of the state transition matrices obtained from MSMs ...')
        spectral_gaps, eig_vals = [analyze_matrix.calc_spectral_gap(mtx) for mtx in mtx_list]
        for i in range(REXEE.n_sim):
            print(f'       - Trajectory {i}: {spectral_gaps[i]:.3f}')

        # 3-6. Calculate the stationary distribution for each trajectory
        print('\n3-6. Calculating the stationary distributions from the transition matrices obtained from MSMs ...')
        pi_list = [m.pi for m in models]
        for i in range(REXEE.n_sim):
            print(f'       - Trajectory {i}: {", ".join([f"{j:.3f}" for j in pi_list[i]])}')
        if len({len(i) for i in pi_list}) == 1:  # all lists in pi_list have the same length
            print(f'       - Average of the above: {", ".join([f"{i:.3f}" for i in np.mean(pi_list, axis=0)])}')

        # 3-7. Calculate the mean first-passage time (MFPT) from each MSM
        print('\n3-7. Calculating the mean first-passage times (MFPTs) ...')
        # note that it's not m.mfpt(min(m.active_set), max(m.active_set)) as the input to mfpt should be indices
        # though sometimes these two could be same.
        mfpt_list = [m.mfpt(0, m.nstates - 1) for m in models]
        for i in range(REXEE.n_sim):
            print(f'       - Trajectory {i}: {mfpt_list[i]:.1f} ps')
        print(f'       - Average of the above: {np.mean(mfpt_list):.1f} ps (std: {np.std(mfpt_list, ddof=1):.1f} ps)')

        # 3-8. Calculate the state index correlation time for each trajectory
        print('\n3-8. Plotting the state index correlation times for all trajectories ...')
        msm_analysis.plot_acf(models, REXEE.n_tot, f'{args.dir}/state_ACF.png')

    # Section 4 (or Section 3). Free energy calculations
    if REXEE.free_energy is True:
        section_idx += 1
        print(f'\n[ Section {section_idx}. Free energy calculations ]')

        if REXEE.modify_coords is False:
            # 4-1. Subsampling the data
            data_list = []   # either a list of u_nk or a list of dhdl
            if REXEE.df_data_type == 'u_nk':
                if os.path.isfile(f'{args.dir}/u_nk_data.pickle') is True:
                    print('Loading the preprocessed data u_nk ...')
                    with open(f'{args.dir}/u_nk_data.pickle', 'rb') as handle:
                        data_all = pickle.load(handle)
                        data_list, t_idx_list, g_list = data_all[0], data_all[1], data_all[2]
            else:  # should always be 'dhdl'
                if os.path.isfile(f'{args.dir}/dHdl_data.pickle') is True:
                    print('Loading the preprocessed data dHdl ...')
                    with open(f'{args.dir}/dHdl_data.pickle', 'rb') as handle:
                        data_all = pickle.load(handle)
                        data_list, t_idx_list, g_list = data_all[0], data_all[1], data_all[2]

            if data_list == []:
                files_list = [natsort.natsorted(glob.glob(f'sim_{i}/iteration_*/*dhdl*xvg')) for i in range(REXEE.n_sim)]  # noqa: E501
                data_list, t_idx_list, g_list = analyze_free_energy.preprocess_data(files_list, REXEE.temp, REXEE.df_data_type, REXEE.df_spacing)  # noqa: E501

                data_all = [data_list, t_idx_list, g_list]
                with open(f'{args.dir}/{REXEE.df_data_type}_data.pickle', 'wb') as handle:
                    pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # 4-2. Calculate the free energy profile
            f, f_err, estimators = analyze_free_energy.calculate_free_energy(data_list, REXEE.state_ranges, REXEE.df_method, REXEE.err_method, REXEE.n_bootstrap, REXEE.seed)  # noqa: E501

            print('Plotting the full-range free energy profile ...')
            analyze_free_energy.plot_free_energy(f, f_err, f'{args.dir}/free_energy_profile.png')

            print('The full-range free energy profile averaged over all replicas:')
            print(f"  {', '.join(f'{f[i]:.3f} +/- {f_err[i]:.3f} kT' for i in range(REXEE.n_tot))}")
            print(f'The free energy difference between the coupled and decoupled states: {f[-1]:.3f} +/- {f_err[-1]:.3f} kT')  # noqa: E501

            if REXEE.df_ref is not None:
                rmse_list = analyze_free_energy.calculate_df_rmse(estimators, REXEE.df_ref, REXEE.state_ranges)
                for i in range(REXEE.n_sim):
                    print(f'RMSE of the free energy profile for alchemical range {i} (states {REXEE.state_ranges[i][0]} to {REXEE.state_ranges[i][-1]}): {rmse_list[i]:.2f} kT')  # noqa: E501

            # 4-3. Recalculate the free energy profile if subsampling_avg is True
            if REXEE.subsampling_avg is True:
                print('\nUsing averaged start index of the equilibrated data and the avearged statistic inefficiency to re-perform free energy calculations ...')  # noqa: E501
                t_avg = int(np.mean(t_idx_list)) + 1   # Using the ceiling function to be a little more conservative
                g_avg = np.array(g_list).prod() ** (1/len(g_list))  # geometric mean
                print(f'Averaged start index: {t_avg}')
                print(f'Averaged statistical inefficiency: {g_avg:.2f}')

                data_list, t_idx_list, g_list = analyze_free_energy.preprocess_data(files_list, REXEE.temp, REXEE.df_data_type, REXEE.df_spacing, t_avg, g_avg)  # noqa: E501
                data_all = [data_list, t_idx_list, g_list]
                with open(f'{args.dir}/{REXEE.df_data_type}_data_avg_subsampling.pickle', 'wb') as handle:
                    pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

                f, f_err, estimators = analyze_free_energy.calculate_free_energy(data_list, REXEE.state_ranges, REXEE.df_method, REXEE.err_method, REXEE.n_bootstrap, REXEE.seed)  # noqa: E501
                print('Plotting the full-range free energy profile ...')
                analyze_free_energy.plot_free_energy(f, f_err, f'{args.dir}/free_energy_profile_avg_subsampling.png')

                print('The full-range free energy profile averaged over all replicas:')
                print(f"  {', '.join(f'{f[i]:.3f} +/- {f_err[i]:.3f} kT' for i in range(REXEE.n_tot))}")
                print(f'The free energy difference between the coupled and decoupled states: {f[-1]:.3f} +/- {f_err[-1]:.3f} kT')  # noqa: E501

                if REXEE.df_ref is not None:
                    rmse_list = analyze_free_energy.calculate_df_rmse(estimators, REXEE.df_ref, REXEE.state_ranges)
                    for i in range(REXEE.n_sim):
                        print(f'RMSE of the free energy profile for alchemical range {i} (states {REXEE.state_ranges[i][0]} to {REXEE.state_ranges[i][-1]}): {rmse_list[i]:.2f} kT')  # noqa: E501

        else:  # MT-REXEE means each simulation is a seperate transformation
            fe_est, fe_err, trans = [], [], []
            for sim in range(REXEE.n_sim):
                print(f'Computing Free Energy for Simulation {sim}')
                # 4-1. Subsampling the data
                data_list = []   # either a list of u_nk or a list of dhdl
                if REXEE.df_data_type == 'u_nk':
                    if os.path.isfile(f'{args.dir}/u_nk_data_{sim}.pickle') is True:
                        print('Loading the preprocessed data u_nk ...')
                        with open(f'{args.dir}/u_nk_data_{sim}.pickle', 'rb') as handle:
                            data_all = pickle.load(handle)
                            data_list, t_idx_list, g_list = data_all[0], data_all[1], data_all[2]
                else:  # should always be 'dhdl'
                    if os.path.isfile(f'{args.dir}/dHdl_data_{sim}.pickle') is True:
                        print('Loading the preprocessed data dHdl ...')
                        with open(f'{args.dir}/dHdl_data_{sim}.pickle', 'rb') as handle:
                            data_all = pickle.load(handle)
                            data_list, t_idx_list, g_list = data_all[0], data_all[1], data_all[2]

                if data_list == []:
                    files_list = [natsort.natsorted(glob.glob(f'sim_{sim}/iteration_*/*dhdl*xvg'))]
                    data_list, t_idx_list, g_list = analyze_free_energy.preprocess_data(files_list, REXEE.temp, REXEE.df_data_type, REXEE.df_spacing)  # noqa: E501

                    data_all = [data_list, t_idx_list, g_list]
                    with open(f'{args.dir}/{REXEE.df_data_type}_data_{sim}.pickle', 'wb') as handle:
                        pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # 4-2. Calculate the free energy profile
                f, f_err, estimators = analyze_free_energy.calculate_free_energy(data_list, REXEE.state_ranges[sim], REXEE.df_method, REXEE.err_method, REXEE.n_bootstrap, REXEE.seed, MTREXEE=True)  # noqa: E501

                print(f'The free energy difference between the coupled and decoupled states: {f:.3f} +/- {f_err:.3f} kT')  # noqa: E501
                fe_est.append(f)
                fe_err.append(f_err)
                trans.append(REXEE.resname_list[sim])
            FE_df = pd.DataFrame({'Transformation': trans, 'FE Estimate': fe_est, 'FE Error': fe_err})
            FE_df.to_csv(f'{args.dir}/FE_estimates.csv')

    # Section 5. Process trajecotries for MT-REXEE
    if REXEE.modify_coords is not None:
        # Section 5.1. Create end-state trajecotries for each simulation
        section_idx += 1
        print(f'\n[ Section {section_idx}. Create end-state trajecotries for each simulation')

        l0, l1, ps_per_frame = gmx_parser.get_end_states(f'{REXEE.working_dir}/sim_0/iteration_0/expanded.mdp')
        n_sim, n_iter = np.shape(rep_trajs)
        if REXEE.swap_rep_pattern is None:
            raise Exception('MT-REXEE trajectory analysis requires swap_rep_pattern to be defined')
        analyze_traj.end_states_only_traj(REXEE.working_dir, n_sim, n_iter, l0, l1, REXEE.swap_rep_pattern, ps_per_frame)  # noqa: E501

        # Section 5.2. Create concatenated trajectories for each individual simulation
        print('5.2. Create concatenated trajectories for each individual simulation')
        analyze_traj.concat_sim_traj(REXEE.working_dir, n_sim, n_iter, REXEE.gro)

        print('5.3. Concatenate dhdl files')
        for s in range(n_sim):
            if os.path.exists(f'{args.dir}/sim_{s}.xvg'):
                continue
            output_file = open(f'{args.dir}/sim_{s}.xvg', 'w')
            for i in range(n_iter):
                input_file = open(f'{REXEE.working_dir}/sim_{s}/iteration_{i}/dhdl.xvg').readlines()
                if i == 0:
                    for line in input_file:
                        output_file.write(line)
                    time_value = float(input_file[-1].split(' ')[0])
                    time_step = np.round(time_value - float(input_file[-2].split(' ')[0]), 4)
                else:
                    skipped_first = False
                    for line in input_file:
                        if line[0] != '#' and line[0] != '@':
                            if skipped_first is False:
                                skipped_first = True
                            else:
                                time_value += time_step
                                time_str = f'{time_value:.4f}'
                                n = len(line.split(' ')[0])
                                new_line = time_str + line[n:]
                                output_file.write(new_line)
            output_file.close()

    # Section 6. Calculate the time spent in GROMACS (This could take a while.)
    t_wall_tot, t_sync, _ = utils.analyze_REXEE_time()
    print(f'\nTotal wall time GROMACS spent to finish all iterations: {utils.format_time(t_wall_tot)}')
    print(f'Total time spent in syncrhonizing all replicas: {utils.format_time(t_sync)}')

    print(f'\nTime spent on data analysis: {utils.format_time(time.time() - t0)}')
