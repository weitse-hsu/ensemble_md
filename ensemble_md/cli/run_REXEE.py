####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################
import os
import sys
import time
import copy
import shutil
import traceback
import argparse
import numpy as np
from mpi4py import MPI
from datetime import datetime

from ensemble_md.utils import utils
from ensemble_md.replica_exchange_EE import ReplicaExchangeEE


def initialize(args):
    parser = argparse.ArgumentParser(
        description='This CLI runs a REXEE simulation given necessary inputs.')
    parser.add_argument('-y',
                        '--yaml',
                        type=str,
                        default='params.yaml',
                        help='The file path of the input YAML file that contains REXEE parameters. \
                            (Default: params.yaml)')
    parser.add_argument('-c',
                        '--ckpt',
                        type=str,
                        default='rep_trajs.npy',
                        help='The file path of the NPY file containing the replica-space trajectories. This file is a \
                            necessary checkpoint file for extending the simulation. (Default: rep_trajs.npy)')
    parser.add_argument('-g',
                        '--g_vecs',
                        type=str,
                        default='g_vecs.npy',
                        help='The file path of the NPY file containing the time series of the whole-range\
                            alchemical weights. This file is a necessary input if one wants to update the \
                            file when extending a weight-updating simulation. (Default: g_vecs.npy)')
    parser.add_argument('-e',
                        '--equil',
                        type=str,
                        default='equil.npy',
                        help='The file path of the NPY file containing the equilibration times for all simulations \
                            when completing a variable weight REXEE simulation. (Default: equil.npy)')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='run_REXEE_log.txt',
                        help='The file path of the output file for logging how replicas interact with each other. \
                            (Default: run_REXEE_log.txt)')
    parser.add_argument('-m',
                        '--maxwarn',
                        type=int,
                        default=0,
                        help='The maximum number of warnings in parameter specification to be ignored. (Default: 0)')
    args_parse = parser.parse_args(args)

    return args_parse


def main():
    t1 = time.time()
    args = initialize(sys.argv[1:])
    sys.stdout = utils.Logger(logfile=args.output)
    sys.stderr = utils.Logger(logfile=args.output)

    # Step 1: Set up MPI rank and instantiate ReplicaExchangeEE to set up REXEE parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Note that this is a GLOBAL variable

    if rank == 0:
        print(f'Current time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        print(f'Command line: {" ".join(sys.argv)}\n')

    REXEE = ReplicaExchangeEE(args.yaml)

    if rank == 0:
        # Print out simulation parameters
        REXEE.print_params()

        # Print out warnings and fail if needed
        for i in REXEE.warnings:
            print(f'\n{i}\n')

        if len(REXEE.warnings) > args.maxwarn:
            print(f"The execution failed due to warning(s) about parameter spcificaiton. Check the warnings, or consider setting maxwarn in the input YAML file if you find them harmless.")  # noqa: E501, F541
            comm.Abort(101)

    # Step 2: If there is no checkpoint file found/provided, perform the 1st iteration (index 0)
    if os.path.isfile(args.ckpt) is False:
        start_idx = 1

        # 2-1. Set up input files for all simulations
        if rank == 0:
            for i in range(REXEE.n_sim):
                os.mkdir(f'{REXEE.working_dir}/sim_{i}')
                os.mkdir(f'{REXEE.working_dir}/sim_{i}/iteration_0')
                MDP = REXEE.initialize_MDP(i)
                MDP.write(f"{REXEE.working_dir}/sim_{i}/iteration_0/{REXEE.mdp}", skipempty=True)
            if REXEE.modify_coords == 'default' and (not os.path.exists('residue_connect.csv') or not os.path.exists('residue_swap_map.csv')):  # noqa: E501
                REXEE.process_top()

        # 2-2. Run the first set of simulations
        REXEE.run_REXEE(0)

    else:
        if rank == 0:
            # If there is a checkpoint file, we see the execution as an extension of an REXEE simulation
            ckpt_data = np.load(args.ckpt)
            start_idx = len(ckpt_data[0])  # The length should be the same for the same axis
            print(f'\nGetting prepared to extend the REXEE simulation from iteration {start_idx} ...')

            if start_idx == REXEE.n_iter:
                print('Extension aborted: The expected number of iterations have been completed!')
                MPI.COMM_WORLD.Abort(1)
            else:
                print('Deleting data generated after the checkpoint ...')
                for i in range(REXEE.n_sim):
                    n_finished = len(next(os.walk(f'{REXEE.working_dir}/sim_{i}'))[1])  # number of finished iterations
                    for j in range(start_idx, n_finished):
                        print(f'  Deleting the folder {REXEE.working_dir}/sim_{i}/iteration_{j}')
                        shutil.rmtree(f'{REXEE.working_dir}/sim_{i}/iteration_{j}')

            # Read g_vecs.npy and rep_trajs.npy so that new data can be appended, if any.
            # Read equil.npy if running variable weight REXEE simulations
            # Note that these two arrays are created in rank 0 and should always be operated in rank 0,
            # or broadcasting is required.
            REXEE.rep_trajs = [list(i) for i in ckpt_data]
            if os.path.isfile(args.g_vecs) is True:
                REXEE.g_vecs = [list(i) for i in np.load(args.g_vecs)]
            if REXEE.fixed_weights is not True and os.path.isfile(args.equil) is True:
                REXEE.equil = np.load(args.equil)
                print(f'equil: {REXEE.equil}')
            if REXEE.proposal == 'random_range' and os.path.isfile('track_swap_frame.npy'):
                REXEE.track_swap_frame = list(np.load('track_swap_frame.npy'))
        else:
            start_idx = None

        start_idx = comm.bcast(start_idx, root=0)  # so that all the ranks are aware of start_idx

    # 2-3. Get the reference distance for the distance restraint specified in the pull code, if any.
    pullx_file = f'{REXEE.working_dir}/sim_0/iteration_0/pullx.xvg'
    REXEE.get_ref_dist(pullx_file)

    for i in range(start_idx, REXEE.n_iter):
        # For a large code block like below executed on rank 0, we try to catch any exception and abort the simulation.
        # So if there is bug, the execution will be terminated and no computation time will be wasted.
        try:
            if rank == 0:
                # Step 3: Swap the coordinates
                # 3-1. For all the replica simulations,
                #   (1) Find the last sampled state and the corresponding lambda values from the DHDL files.
                #   (2) Find the final Wang-Landau incrementors and weights from the LOG files.
                dhdl_files = [f'{REXEE.working_dir}/sim_{j}/iteration_{i - 1}/dhdl.xvg' for j in range(REXEE.n_sim)]
                log_files = [f'{REXEE.working_dir}/sim_{j}/iteration_{i - 1}/md.log' for j in range(REXEE.n_sim)]
                states_ = REXEE.extract_final_dhdl_info(dhdl_files)
                wl_delta, weights_, counts_ = REXEE.extract_final_log_info(log_files)
                print()

                # 3-2. Identify swappable pairs, propose swap(s), calculate P_acc, and accept/reject swap(s)
                # Note after `get_swapping_pattern`, `states_` and `weights_` won't be necessarily
                # since they are updated by `get_swapping_pattern`. (Even if the function does not explicitly
                # returns `states_` and `weights_`, `states_` and `weights_` can still be different after
                # the use of the function.) Therefore, here we create copies for `states_` and `weights_`
                # before the use of `get_swapping_pattern`, so we can use them in `weight_correction`,
                # `combine_weights` and `update_MDP`.
                states = copy.deepcopy(states_)
                weights = copy.deepcopy(weights_)
                counts = copy.deepcopy(counts_)
                swap_pattern, swap_list, swap_index, states = REXEE.get_swapping_pattern(dhdl_files, states_, i)  # swap_list will only be used for modify_coords  # noqa: E501
                print(f'swap pattern: {swap_pattern}')

                # 3-3. Perform weight correction/weight combination
                if wl_delta != [None for i in range(REXEE.n_sim)]:  # weight-updating
                    print(f'\nCurrent Wang-Landau incrementors: {wl_delta}\n')

                # (1) First we prepare the time-averaged weights to be combined, if needed.
                # Note that although averaged weights are sometimes used for weight correction/weight combination,
                # the final weights are always used for calculating the acceptance ratio.
                if REXEE.N_cutoff != -1 or REXEE.w_combine is True:
                    # Only when weight correction/weight combination is needed.
                    weights_avg, weights_err = REXEE.get_averaged_weights(log_files)

                    # Calculate the RMSE between the averaged weights and the final weights by the way.
                    rmse_list = [utils.calc_rmse(weights_avg[i], weights[i]) for i in range(REXEE.n_sim)]
                    rmse_str = ', '.join([f'{i:.2f}' for i in rmse_list])
                    print(f'RMSE between the final weights and time-averaged weights for each replica: {rmse_str} kT')

                # (2) Now we perform weight correction/weight combination.
                # The product of this step should always be named as "weights" to be used in update_MDP
                if REXEE.N_cutoff != -1 and REXEE.w_combine is True:
                    # Perform both weight correction and weight combination
                    if REXEE.verbose is True:
                        print('Performing weight correction ...')
                    else:
                        print('Performing weight correction ...', end='')
                    weights_preprocessed = REXEE.weight_correction(weights_avg, counts)

                    if REXEE.verbose is True:
                        print('Performing weight combination ...')
                    else:
                        print('Performing weight combination ...', end='')
                    if REXEE.w_mean_type == 'simple':
                        weights, g_vec = REXEE.combine_weights(weights_preprocessed)  # simple means
                    else:
                        # Note that here weights_err are acutally not the uncertainties for weights_prepocessed
                        # but weights_avg ... We might need to disable this feature in the future.
                        weights, g_vec = REXEE.combine_weights(weights_preprocessed, weights_err)  # inverse-variance weighting  # noqa: E501
                    REXEE.g_vecs.append(g_vec)

                    # Check if histogram correction is needed after weight combination
                    if REXEE.hist_corr is True:
                        print('Performing histogram correction ...')
                        counts = REXEE.histogram_correction(counts_)
                    else:
                        print('Note: No histogram correction will be performed.')

                elif REXEE.N_cutoff == -1 and REXEE.w_combine is True:
                    # Only perform weight combination
                    print('Note: No weight correction will be performed.')
                    if REXEE.verbose is True:
                        print('Performing weight combination ...')
                    else:
                        print('Performing weight combination ...', end='')
                    if REXEE.w_mean_type == 'simple':
                        weights, g_vec = REXEE.combine_weights(weights_avg)  # simple means
                    else:
                        weights, g_vec = REXEE.combine_weights(weights_avg, weights_err)  # inverse-variance weighting
                    REXEE.g_vecs.append(g_vec)

                    # Check if histogram correction is needed after weight combination
                    if REXEE.hist_corr is True:
                        print('Performing histogram correction ...')
                        counts = REXEE.histogram_correction(counts_)
                    else:
                        print('Note: No histogram correction will be performed.')

                elif REXEE.N_cutoff != -1 and REXEE.w_combine is False:
                    # Only perform weight correction
                    print('Note: No weight combination will be performed.')
                    if REXEE.verbose is True:
                        print('Performing weight correction ...')
                    else:
                        print('Performing weight correction ...', end='')
                    weights = REXEE.weight_correction(weights_avg, counts)
                    _ = REXEE.combine_weights(weights, print_values=False)[1]  # just to print the combined weights  # noqa: E501
                else:
                    print('Note: No weight correction will be performed.')
                    print('Note: No weight combination will be performed.')
                    # Note that in this case, the final weights will be used in the next iteration.
                    _ = REXEE.combine_weights(weights, print_values=False)[1]  # just to print the combiend weights  # noqa: E501

                # 3-4. Modify the MDP files and swap out the GRO files (if needed)
                # Here we keep the lambda range set in mdp the same across different iterations in the same folder but swap out the gro file  # noqa: E501
                # Note we use states (copy of states_) instead of states_ in update_MDP.
                for j in list(range(REXEE.n_sim)):
                    os.mkdir(f'{REXEE.working_dir}/sim_{j}/iteration_{i}')
                    if REXEE.fixed_weights is True:
                        counts = None    # So that this should work also for GROMACS version < 2022.5
                    MDP = REXEE.update_MDP(f"sim_{j}/iteration_{i - 1}/{REXEE.mdp}", j, i, states, wl_delta, weights, counts)   # modify with a new template  # noqa: E501
                    MDP.write(f"{REXEE.working_dir}/sim_{j}/iteration_{i}/{REXEE.mdp}", skipempty=True)
                    # In run_REXEE(i, swap_pattern), where the tpr files will be generated, we use the top file at the
                    # level of the simulation (the file that will be shared by all simulations). For the gro file, we
                    # pass swap_pattern to the function to figure it out internally.

                if REXEE.proposal == 'random_range':
                    # 3-5. Keep track of the frames used for swapping in each trajectory
                    track_frame = np.full(REXEE.n_sim, -1)
                    for s in range(len(swap_list)):
                        swap = swap_list[s]
                        track_frame[swap[0]] = swap_index[s][0]
                        track_frame[swap[1]] = swap_index[s][1]
                    REXEE.track_swap_frame.append(track_frame)
            else:
                swap_pattern, swap_list = None, None

        except Exception:
            print('\n--------------------------------------------------------------------------\n')
            print(f'An error occurred on rank 0:\n{traceback.format_exc()}')
            MPI.COMM_WORLD.Abort(1)

        if -1 not in REXEE.equil and 0 not in REXEE.equil:
            # This is the case where the weights are equilibrated in a weight-updating simulation.
            # As a remidner, REXEE.equil should be a list of 0 after extract_final_log_info in a
            # fixed-weight simulation, and a list of -1 for a weight-updating simulation with unequilibrated weights.
            print('\nSimulation terminated: The weights have been equilibrated for all replicas.')  # this will only be printed in rank 0  # noqa: E501

            # Note that REXEE.equil is avaiable for all ranks but only updated in rank 0. So the if condition here
            # can only be satisfied in rank 0. We broadcast exit_loop to all ranks so that all ranks can exit the
            # simulation at the same time, if the weights get equilibrated.
            exit_loop = True
            comm.bcast(exit_loop, root=0)
            break
        else:
            # Receive exit_loop from rank 0. If exit_loop is True, all ranks will exit the simulation.
            exit_loop = comm.bcast(False, root=0)
            if exit_loop:
                break

        # Step 4: Perform another iteration
        # 4-1. Modify the coordinates of the output gro files if needed.
        swap_pattern = comm.bcast(swap_pattern, root=0)
        swap_list = comm.bcast(swap_list, root=0)

        if len(swap_list) == 0:
            pass
        else:
            if REXEE.modify_coords_fn is not None:
                try:
                    if rank == 0:
                        # If previous swaps were performed undo them
                        for j in range(REXEE.n_sim):
                            gro_backup = f'{REXEE.working_dir}/sim_{swap_list[j][0]}/iteration_{i-1}/confout_backup.gro'
                            gro = f'{REXEE.working_dir}/sim_{swap_list[j][0]}/iteration_{i-1}/confout.gro'
                            if os.path.exists(gro_backup):
                                os.rename(gro_backup, gro)
                                
                        for j in range(len(swap_list)):
                            print('\nModifying the coordinates of the following output GRO files ...')
                            # gro_1 and gro_2 are the simlation outputs (that we want to back up) and the inputs to modify_coords  # noqa: E501
                            gro_1 = f'{REXEE.working_dir}/sim_{swap_list[j][0]}/iteration_{i-1}/confout.gro'
                            gro_2 = f'{REXEE.working_dir}/sim_{swap_list[j][1]}/iteration_{i-1}/confout.gro'
                            print(f'  - {gro_1}\n  - {gro_2}')

                            # Now we rename gro_1 and gro_2 to back them up
                            gro_1_backup = gro_1.split('.gro')[0] + '_backup.gro'
                            gro_2_backup = gro_2.split('.gro')[0] + '_backup.gro'
                            os.rename(gro_1, gro_1_backup)
                            os.rename(gro_2, gro_2_backup)

                            # Here we input gro_1_backup and gro_2_backup and modify_coords_fn will save the modified gro files as gro_1 and gro_2  # noqa: E501
                            REXEE.modify_coords_fn(gro_1_backup, gro_2_backup, swap_index[j])  # the order should not matter  # noqa: E501
                except Exception:
                    print('\n--------------------------------------------------------------------------\n')
                    print(f'\nAn error occurred on rank 0:\n{traceback.format_exc()}')
                    MPI.COMM_WORLD.Abort(1)

        # 4-2. Run another set of simulations
        REXEE.run_REXEE(i, swap_pattern)

        # 4-3. Save data
        if rank == 0:
            if (i + 1) % REXEE.n_ckpt == 0:
                if len(REXEE.g_vecs) != 0:
                    # Save g_vec as a function of time if weight combination was used.
                    np.save(args.g_vecs, REXEE.g_vecs)

                print('\n----- Saving .npy files to checkpoint the simulation ---')
                np.save(args.ckpt, REXEE.rep_trajs)
                if REXEE.fixed_weights is not True:
                    np.save(args.equil, REXEE.equil)
                if REXEE.proposal == 'random_range':
                    np.save('track_swap_frame.npy', REXEE.track_swap_frame)

    # Save the npy files at the end of the simulation anyway.
    if rank == 0:
        if len(REXEE.g_vecs) != 0:  # The length will be 0 only if there is no weight combination.
            np.save(args.g_vecs, REXEE.g_vecs)
        np.save(args.ckpt, REXEE.rep_trajs)
        if REXEE.fixed_weights is not True:
            np.save(args.equil, REXEE.equil)

    # Step 5: Write a summary for the simulation ensemble
    if rank == 0:
        print('\nSummary of the simulation ensemble')
        print('==================================')
        print('Simulation status:')
        for i in range(REXEE.n_sim):
            if REXEE.fixed_weights is True:
                print(f'- Rep {i}: The weights were fixed throughout the simulation.')
            elif REXEE.equil[i] == -1:
                print(f'  - Rep {i}: The weights have not been equilibrated.')
            else:
                idx = int(np.floor(REXEE.equil[i] / (REXEE.dt * REXEE.nst_sim)))
                if REXEE.equil[i] > 1000:
                    units = 'ns'
                    REXEE.equil[i] /= 1000
                else:
                    units = 'ps'
                print(f'  - Rep {i}: The weights have been equilibrated at {REXEE.equil[i]:.2f} {units} (iteration {idx}).')  # noqa: E501

        print(f'\n{REXEE.n_empty_swappable} out of {REXEE.n_iter}, or {REXEE.n_empty_swappable / REXEE.n_iter * 100:.1f}% iterations had an empty list of swappable pairs.')  # noqa: E501
        if REXEE.n_swap_attempts != 0:
            print(f'{REXEE.n_rejected} out of {REXEE.n_swap_attempts}, or {REXEE.n_rejected / REXEE.n_swap_attempts * 100:.1f}% of attempted exchanges were rejected.')  # noqa: E501

        print(f'\nTime elapsed: {utils.format_time(time.time() - t1)}')

    MPI.Finalize()
