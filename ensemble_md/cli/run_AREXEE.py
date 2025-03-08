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
import copy
import time
import traceback
import warnings
from mpi4py import MPI
from datetime import datetime

from ensemble_md.utils import utils
from ensemble_md.cli.run_REXEE import initialize
from ensemble_md.replica_exchange_EE import ReplicaExchangeEE

warnings.warn('This module is only for experimental purposes and still in progress. Please do not use it for any production research.', UserWarning)  # noqa: E501

"""
Currently, this CLI still uses MPI to run REXEE simulations, but it tries to mock some behaviors of asynchronous REXEE
in the following way:
1. Finish an iteration of the REXEE simulation.
2. Based on the time it took for each simulation to finish, figure out the order in which the replicas should be
   added to the queue.
3. Apply a queueing algorithm to figure out what replicas to swap first.

Eventually, we would like to get rid of the use of MPI and really rely on asynchronous parallelization schemes.
The most likely direction is to use functionalities in airflowHPC to manage the queueing and launching of
replicas. If possible, this CLI should be integrated into the CLI run_REXEE.
"""


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

    # Note that here we assume no checkpoint files just to minimize this CLI.
    # We also leave out Step 2-3 since we won't be using this CLI to test calculations with any restraints.
    start_idx = 1

    # 2-1. Set up input files for all simulations
    if rank == 0:
        for i in range(REXEE.n_sim):
            os.mkdir(f'{REXEE.working_dir}/sim_{i}')
            os.mkdir(f'{REXEE.working_dir}/sim_{i}/iteration_0')
            MDP = REXEE.initialize_MDP(i)
            MDP.write(f"{REXEE.working_dir}/sim_{i}/iteration_0/expanded.mdp", skipempty=True)
        if REXEE.modify_coords == 'default' and (not os.path.exists('residue_connect.csv') or not os.path.exists('residue_swap_map.csv')):  # noqa: E501
            REXEE.process_top()

    # 2-2. Run the first set of simulations
    REXEE.run_REXEE(0)

    for i in range(start_idx, REXEE.n_iter):
        try:
            if rank == 0:
                # (New) This is where we use the queueing algorithm
                


                # Step 3: Swap the coordinates
                # Note that here we leave out Steps 3-3 and 3-4, which are for weight combination/correction and
                # coordinate modification, respectively.

                # 3-1. Extract the final dhdl and log files from the previous iteration
                dhdl_files = [f'{REXEE.working_dir}/sim_{j}/iteration_{i - 1}/dhdl.xvg' for j in range(REXEE.n_sim)]
                log_files = [f'{REXEE.working_dir}/sim_{j}/iteration_{i - 1}/md.log' for j in range(REXEE.n_sim)]
                states_ = REXEE.extract_final_dhdl_info(dhdl_files)
                wl_delta, weights_, counts_ = REXEE.extract_final_log_info(log_files)
                print()

                # 3-2. Identify swappable pairs, propose swap(s), calculate P_acc, and accept/reject swap(s)
                states = copy.deepcopy(states_)  # noqa: F841
                weights = copy.deepcopy(weights_)  # noqa: F841
                counts = copy.deepcopy(counts_)  # noqa: F841
                swap_pattern, swap_list = REXEE.get_swapping_pattern(dhdl_files, states_)  # swap_list will only be used for modify_coords  # noqa: E501
            else:
                swap_pattern, swap_list = None, None  # noqa: F841

        except Exception:
            print('\n--------------------------------------------------------------------------\n')
            print(f'An error occurred on rank 0:\n{traceback.format_exc()}')
            MPI.COMM_WORLD.Abort(1)

            # Note that we leave out the block for exiting the for loop when the weights got equilibrated, as this CLI
            # won't be tested for weight-updating simulations for now.

            # Step 4: Perform another iteration
            # Here we leave out the block that uses swap_list, which is only for coordinate modifications.
            swap_pattern = comm.bcast(swap_pattern, root=0)

            # Here we run another set of simulations (i.e. Step 4-2 in CLI run_REXEE)
            REXEE.run_REXEE(i, swap_pattern)

            # Here we leave out the block for saving data (i.e. Step 4-3 in CLI run_REXEE) since we won't
            # run for too many iterations when testing this CLI.

            # Step 5: Write a summary for the simulation ensemble
            if rank == 0:
                print('\nSummary of the simulation ensemble')
                print('==================================')

                # We leave out the section showing the simulation status.
                print(f'\n{REXEE.n_empty_swappable} out of {REXEE.n_iter}, or {REXEE.n_empty_swappable / REXEE.n_iter * 100:.1f}% iterations had an empty list of swappable pairs.')  # noqa: E501
                if REXEE.n_swap_attempts != 0:
                    print(f'{REXEE.n_rejected} out of {REXEE.n_swap_attempts}, or {REXEE.n_rejected / REXEE.n_swap_attempts * 100:.1f}% of attempted exchanges were rejected.')  # noqa: E501

                print(f'\nTime elapsed: {utils.format_time(time.time() - t1)}')

            MPI.Finalize()
