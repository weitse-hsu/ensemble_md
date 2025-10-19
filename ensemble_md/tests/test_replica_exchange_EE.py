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
Unit tests for the module replica_exchange_EE.py.
"""
import re
import os
import sys
import yaml
import copy
import random
import pytest
import subprocess
import numpy as np
import ensemble_md
from unittest.mock import patch
from ensemble_md.utils import gmx_parser
from ensemble_md.replica_exchange_EE import ReplicaExchangeEE
from ensemble_md.utils.exceptions import ParameterError

import logging
logging.basicConfig(level=logging.DEBUG)

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


@pytest.fixture
def params_dict():
    """
    Generates a dictionary containing the required REXEE parameters.
    """
    REXEE_dict = {
        'gmx_executable': 'gmx',
        'gro': 'ensemble_md/tests/data/sys.gro',
        'top': 'ensemble_md/tests/data/sys.top',
        'mdp': 'ensemble_md/tests/data/expanded.mdp',
        'n_sim': 4,
        'n_iter': 10,
        's': 1,
    }
    yield REXEE_dict

    # Remove the file after the unit test is done.
    if os.path.isfile('params.yaml') is True:
        os.remove('params.yaml')


def get_REXEE_instance(input_dict, yml_file='params.yaml'):
    """
    Saves a dictionary as a yaml file and use it to instantiate the ReplicaExchangeEE class.
    """
    with open(yml_file, 'w') as f:
        yaml.dump(input_dict, f)
    REXEE = ReplicaExchangeEE(yml_file)
    return REXEE


def check_param_error(REXEE_dict, param, match, wrong_val='cool', right_val=None):
    """
    Test if ParameterError is raised if a parameter is not well-defined.
    """
    REXEE_dict[param] = wrong_val
    with pytest.raises(ParameterError, match=match):
        get_REXEE_instance(REXEE_dict)
    REXEE_dict[param] = right_val  # so that REXEE_dict can be read without failing in later assertions

    return REXEE_dict


class Test_ReplicaExchangeEE:
    def test_init(self, params_dict):
        logging.debug("Initializing ReplicaExchangeEE")
        REXEE = get_REXEE_instance(params_dict)
        assert REXEE.yaml == 'params.yaml'
        logging.debug("Finished initializing ReplicaExchangeEE")

    def test_set_params_error(self, params_dict):
        # 1. Required parameters
        del params_dict['n_sim']
        with pytest.raises(ParameterError, match="Required parameter 'n_sim' not specified in params.yaml."):
            get_REXEE_instance(params_dict)
        params_dict['n_sim'] = 4  # so params_dict can be read without failing in the assertions below

        # 2. Available options
        check_param_error(params_dict, 'proposal', "The specified proposal scheme is not available. Available options include 'single', 'neighboring', 'exhaustive', 'random_range'.", 'cool', 'exhaustive')  # noqa: E501
        check_param_error(params_dict, 'df_method', "The specified free energy estimator is not available. Available options include 'TI', 'BAR', and 'MBAR'.")  # noqa: E501
        check_param_error(params_dict, 'err_method', "The specified method for error estimation is not available. Available options include 'propagate', and 'bootstrap'.")  # noqa: E501

        # 3. Integer parameters
        check_param_error(params_dict, 'nst_sim', "The parameter 'nst_sim' should be an integer.")
        check_param_error(params_dict, 'seed', "The parameter 'seed' should be an integer.")
        check_param_error(params_dict, 'n_sim', "The parameter 'n_sim' should be an integer.", 4.1, 4)

        # 4. Positive parameters
        check_param_error(params_dict, 'nst_sim', "The parameter 'nst_sim' should be positive.", -5, 500)
        check_param_error(params_dict, 'n_iter', "The parameter 'n_iter' should be positive.", 0, 10)

        # 5. Non-negative parameters
        check_param_error(params_dict, 'N_cutoff', "The parameter 'N_cutoff' should be non-negative unless no weight correction is needed, i.e. N_cutoff = -1.", -5)  # noqa: E501

        # 6. String parameters
        check_param_error(params_dict, 'mdp', "The parameter 'mdp' should be a string.", 3, 'ensemble_md/tests/data/expanded.mdp')  # noqa: E501

        # 7. Boolean parameters
        check_param_error(params_dict, 'msm', "The parameter 'msm' should be a boolean variable.", 3, False)

        # 8. Errors related to nstlog and nstdhdl
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))  # A perfect mdp file

        # 8-1. nstlog is not a factor of nst_sim
        mdp['nstlog'] = 200
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'
        params_dict['nst_sim'] = 100
        with pytest.raises(ParameterError, match='The parameter "nstlog" must be a factor of the parameter "nst_sim" specified in the YAML file.'):  # noqa: E501
            get_REXEE_instance(params_dict)

        # 8-2. nstdhdl is not a factor of nst_sim
        mdp['nstlog'] = 100
        mdp['nstdhdl'] = 200
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        with pytest.raises(ParameterError, match='The parameter "nstdhdl" must be a factor of the parameter "nst_sim" specified in the YAML file.'):  # noqa: E501
            get_REXEE_instance(params_dict)

        # 8-3. nstexpanded is not a factor of nstdhdl
        mdp['nstdhdl'] = 100
        mdp['nstexpanded'] = 50
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        with pytest.raises(ParameterError, match='In REXEE, the parameter "nstdhdl" must be a factor of the parameter "nstexpanded", or the calculation of acceptance ratios may be wrong.'):  # noqa: E501
            get_REXEE_instance(params_dict)

        os.remove(os.path.join(input_path, "expanded_test.mdp"))

        # 9. n_sub < 1
        params_dict['s'] = 5
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded.mdp'
        # Note that the parentheses are special characters that need to be escaped in regular expressions
        with pytest.raises(ParameterError, match=r'There must be at least two states for each replica \(current value: -6\). The current specified configuration \(n_tot=9, n_sim=4, s=5\) does not work for REXEE.'):  # noqa: E501
            get_REXEE_instance(params_dict)

        # 10. s < 0
        params_dict['s'] = -1
        with pytest.raises(ParameterError, match="The parameter 's' should be non-negative."):
            get_REXEE_instance(params_dict)
        params_dict['s'] = 1  # set back to a normal value

        # 11. Cases for MT-REXEE (relevant parameters: add_swappables, modify_coords, etc.)
        params_dict['gro'] = ['ensemble_md/tests/data/sys.gro']
        with pytest.raises(ParameterError, match="The number of the input GRO files must be the same as the number of replicas, if multiple are specified."):  # noqa: E501
            get_REXEE_instance(params_dict)
        params_dict['gro'] = 'ensemble_md/tests/data/sys.gro'  # set back to a normal value

        # 12. Test mdp_args
        params_dict['mdp_args'] = 3
        with pytest.raises(ParameterError, match="The parameter 'mdp_args' should be a dictionary."):
            get_REXEE_instance(params_dict)

        params_dict['mdp_args'] = {'ref_p': 1.0}
        with pytest.raises(ParameterError, match="The values specified in 'mdp_args' should be lists."):
            get_REXEE_instance(params_dict)

        params_dict['mdp_args'] = {'ref_p': [1.0, 1.0, 1.0, 1.0]}
        with pytest.raises(ParameterError, match="MDP parameters set by 'mdp_args' should differ across at least two replicas."):  # noqa: E501
            get_REXEE_instance(params_dict)

        params_dict['mdp_args'] = {5: [1, 1, 1, 1.01]}  # set back to a normal value
        with pytest.raises(ParameterError, match="All keys specified in 'mdp_args' should be strings."):
            get_REXEE_instance(params_dict)

        params_dict['mdp_args'] = {'ref-p': [1.0, 1.01, 1.02, 1.03]}
        with pytest.raises(ParameterError, match="ensemble_md convention: Parameters specified in 'mdp_args' must use underscores in place of hyphens."):  # noqa: E501
            get_REXEE_instance(params_dict)

        params_dict['mdp_args'] = {'ref_p': [1.0, 1.01, 1.02]}
        with pytest.raises(ParameterError, match="The number of values specified for each key in 'mdp_args' should be the same as the number of replicas."):  # noqa: E501
            get_REXEE_instance(params_dict)

        params_dict['mdp_args'] = {'nstlog': [200, 100, 100, 100]}
        with pytest.raises(ParameterError, match='The parameter "nstlog" must be a factor of the parameter "nst_sim" specified in the YAML file.'):  # noqa: E501
            get_REXEE_instance(params_dict)

        params_dict['mdp_args'] = {'nstdhdl': [200, 100, 100, 100]}
        with pytest.raises(ParameterError, match='The parameter "nstdhdl" must be a factor of the parameter "nst_sim" specified in the YAML file.'):  # noqa: E501
            get_REXEE_instance(params_dict)

        params_dict['mdp_args'] = {'nstdhdl': [20, 10, 10, 10], 'nstexpanded': [10, 50, 10, 10]}
        with pytest.raises(ParameterError, match='In REXEE, the parameter "nstdhdl" must be a factor of the parameter "nstexpanded", or the calculation of acceptance ratios may be wrong.'):  # noqa: E501
            get_REXEE_instance(params_dict)
        params_dict['mdp_args'] = None  # set back to a normal value

        # 13. Test the parameter 'pull'
        params_dict['nst_sim'] = 2000
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded_pull.mdp"))
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'

        mdp['pull_coord1_geometry'] = 'direction'
        mdp.write(params_dict['mdp'])
        get_REXEE_instance(params_dict)

        mdp['pull_coord1_geometry'] = 'distance'  # set back to the original value
        mdp['pull_coord1_start'] = 'no'
        mdp.write(params_dict['mdp'])
        get_REXEE_instance(params_dict)

        mdp['pull_coord1_start'] = 'yes'
        mdp.write(params_dict['mdp'])
        mdp['pull_nstxout'] = 0
        mdp.write(params_dict['mdp'])
        REXEE_pull = get_REXEE_instance(params_dict)
        warning = 'A non-zero value should be specified for pull_nstxout if pull_coord*_start is set to yes.'
        assert warning in REXEE_pull.warnings

        os.remove(os.path.join(input_path, "expanded_test.mdp"))

    def test_set_params_warnings(self, params_dict):
        # 1. Non-recognizable parameter in the YAML file
        params_dict['cool'] = 10
        REXEE = get_REXEE_instance(params_dict)
        warning = 'Warning: Parameter "cool" specified in the input YAML file is not recognizable.'
        assert warning in REXEE.warnings

        # 2. Warnings related to the mdp file
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))  # A perfect mdp file
        mdp['lmc_seed'] = 1000
        mdp['gen_seed'] = 1000
        mdp['wl_scale'] = ''
        mdp['gen_vel'] = 'no'
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))

        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'
        params_dict['N_cutoff'] = 1000
        REXEE = get_REXEE_instance(params_dict)

        warning_1 = 'Warning: The weight correction/weight combination method is specified but will not be used since the weights are fixed.'  # noqa: E501
        warning_2 = 'Warning: We recommend setting lmc_seed as -1 so the random seed is different for each iteration.'
        warning_3 = 'Warning: We recommend setting gen_seed as -1 so the random seed is different for each iteration.'
        warning_4 = 'Warning: We recommend generating new velocities for each iteration to avoid potential issues with the detailed balance.'  # noqa: E501

        assert warning_1 in REXEE.warnings
        assert warning_2 in REXEE.warnings
        assert warning_3 in REXEE.warnings
        assert warning_4 in REXEE.warnings

        os.remove(os.path.join(input_path, "expanded_test.mdp"))

        # 2. Warnings related to the mdp file (for cases where mdp_args is not None)
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))  # A perfect mdp file
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))

        params_dict['mdp_args'] = {'lmc_seed': [-1, -1, -1, 0], 'gen_seed': [-1, -1, -1, 0], 'gen_vel': ['no', 'no', 'no', 'yes']}  # noqa: E501
        REXEE_2 = get_REXEE_instance(params_dict)
        assert warning_2 in REXEE_2.warnings
        assert warning_3 in REXEE_2.warnings
        assert warning_4 in REXEE_2.warnings

    def test_set_params(self, params_dict):
        # 0. Get an REXEE instance to test
        REXEE = get_REXEE_instance(params_dict)

        # 1. Check the required REXEE parameters
        assert REXEE.gmx_executable == 'gmx'
        assert REXEE.gro == "ensemble_md/tests/data/sys.gro"
        assert REXEE.top == "ensemble_md/tests/data/sys.top"
        assert REXEE.mdp == "ensemble_md/tests/data/expanded.mdp"
        assert REXEE.n_sim == 4
        assert REXEE.n_iter == 10
        assert REXEE.s == 1

        # 2. Check the default values of the parameters not specified in params.yaml
        assert REXEE.proposal == "exhaustive"
        assert REXEE.w_combine is False
        assert REXEE.N_cutoff == -1
        assert REXEE.verbose is True
        assert REXEE.runtime_args is None
        assert REXEE.n_ckpt == 100
        assert REXEE.msm is False
        assert REXEE.free_energy is False
        assert REXEE.df_spacing == 1
        assert REXEE.df_method == 'MBAR'
        assert REXEE.err_method == 'propagate'
        assert REXEE.n_bootstrap == 50
        assert REXEE.seed is None

        # 3. Check the MDP parameters
        assert REXEE.dt == 0.002
        assert REXEE.temp == 298
        assert REXEE.nst_sim == 500
        assert REXEE.fixed_weights is False

        # 4. Checked the derived parameters
        # Note that lambda_dict will also be tested in test_map_lambda2state.
        k = 1.380649e-23
        NA = 6.0221408e23
        assert REXEE.kT == k * NA * 298 / 1000
        assert REXEE.lambda_types == ['coul_lambdas', 'vdw_lambdas']
        assert REXEE.n_tot == 9
        assert REXEE.n_sub == 6
        assert REXEE.state_ranges == [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8], ]
        assert REXEE.equil == [-1, -1, -1, -1]  # will be zeros after the first iteration if the weights are fixed
        assert REXEE.n_rejected == 0
        assert REXEE.n_swap_attempts == 0
        assert REXEE.n_empty_swappable == 0
        assert REXEE.rep_trajs == [[0], [1], [2], [3]]

        params_dict['df_method'] = 'MBAR'
        REXEE = get_REXEE_instance(params_dict)
        assert REXEE.df_data_type == 'u_nk'

        params_dict['df_method'] = 'BAR'
        REXEE = get_REXEE_instance(params_dict)
        assert REXEE.df_data_type == 'dhdl'

        params_dict['grompp_args'] = {'-maxwarn': 1}
        params_dict['runtime_args'] = {'-nt': 16, '-ntomp': 8}
        REXEE = get_REXEE_instance(params_dict)
        assert REXEE.runtime_args == {'-nt': 16, '-ntomp': 8}
        assert REXEE.grompp_args == {'-maxwarn': 1}

        assert REXEE.reformatted_mdp is False

    def test_set_params_edge_cases(self, params_dict):
        # In the previous unit tests, we have tested the case where fixed_weights is False. Here we test the others.
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))

        # 1. wl_scale is not specified
        del mdp['wl_scale']
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'
        REXEE = get_REXEE_instance(params_dict)
        assert REXEE.fixed_weights is True

        # 2. wl_scale is not empty
        mdp['wl_scale'] = ''  # It will become an empty array array([]) after read by gmx_parser.abs(x)
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        REXEE = get_REXEE_instance(params_dict)
        assert REXEE.fixed_weights is True
        assert REXEE.modify_coords_fn is None  # Just an additional test for modify_coords_fn

    def test_set_params_mtrexee(self, params_dict):
        # Test 1: Below we check if the parameter "add_swappables" is well-defined.
        params_dict['add_swappables'] = 5
        params_dict['mdp'] = 'expanded.mdp'  # irrelevant to MT-REXEE but just to cover some lines
        with pytest.raises(ParameterError, match="The parameter 'add_swappables' should be a list."):
            get_REXEE_instance(params_dict)

        params_dict['add_swappables'] = [15, 16]
        with pytest.raises(ParameterError, match="The parameter 'add_swappables' should be a nested list."):
            get_REXEE_instance(params_dict)

        params_dict['add_swappables'] = [[-3, 1], [4, 5]]
        with pytest.raises(ParameterError, match="Each number specified in 'add_swappables' should be a non-negative integer."):  # noqa: E501
            get_REXEE_instance(params_dict)

        params_dict['add_swappables'] = [[1, 2, 3], [4, 5]]
        with pytest.raises(ParameterError, match="Each sublist in 'add_swappables' should contain two integers."):  # noqa: E501
            get_REXEE_instance(params_dict)

        # Test 2: Below are some checks for the parameter "modify_coords"
        # 2-1. The case where a function has the same name as the module.
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded.mdp'
        params_dict['modify_coords'] = 'ensemble_md/tests/data/edit_gro.py'
        params_dict['add_swappables'] = [[2, 3], [4, 5]]
        with open('ensemble_md/tests/data/edit_gro.py', 'w') as f:
            f.write('def edit_gro():\n')
            f.write('    pass\n')
        REXEE = get_REXEE_instance(params_dict)
        assert REXEE.modify_coords_fn.__name__ == 'edit_gro'
        os.remove('ensemble_md/tests/data/edit_gro.py')

        # 2-2. The case where no function has the same name as the module.
        params_dict['modify_coords'] = 'ensemble_md/tests/data/check_gro.py'
        with open('ensemble_md/tests/data/check_gro.py', 'w') as f:
            f.write('def test_gro():\n')
            f.write('    pass\n')
        # Below we escape the error message since things like "." could be interpreted as special characters
        err_msg = re.escape("The module for coordinate manipulation (specified through the parameter) must have a function with the same name as the module, i.e., check_gro.")  # noqa: E501
        with pytest.raises(ParameterError, match=err_msg):  # noqa: E501
            REXEE = get_REXEE_instance(params_dict)
        os.remove('ensemble_md/tests/data/check_gro.py')

        # 2-2. The case where we use the dafult function
        params_dict['modify_coords'] = 'default'
        params_dict['swap_rep_pattern'] = [[[0, 1], [1, 0]], [[[1, 1], [2, 0]]]]
        params_dict['resname_list'] = ['A', 'B']
        REXEE = get_REXEE_instance(params_dict)
        assert REXEE.modify_coords_fn.__name__ == 'default_coords_fn'

    @patch('ensemble_md.replica_exchange_EE.subprocess.run')
    @patch('builtins.print')
    def test_check_gmx_executable(self, mock_print, mock_run, params_dict):
        # Here we test the case where the GROMACS executable is not available or an unexpected error occurs.
        # The case where the executable is found is tested in the other unit test.
        # Note that in check_gmx_executable, exceptions are caught (i.e., not raised), with a message printed,
        # so we can only check the printed messages.

        # Test 1: The case where the GROMACS executable is not available
        mock_run.side_effect = subprocess.CalledProcessError(1, ['which', 'gmx'])
        get_REXEE_instance(params_dict)
        mock_run.assert_called()
        mock_print.assert_called_with("gmx is not available.")

        # Test 2: The case where an unexpected error occurs
        mock_run.reset_mock()
        mock_print.reset_mock()

        mock_run.side_effect = Exception("Some error")
        get_REXEE_instance(params_dict)
        mock_run.assert_called()
        mock_print.assert_called_with("An error occurred:\nSome error")

    def test_reformat_MDP(self, params_dict):
        # Note that the function reformat_MDP is called in set_params
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))
        mdp['cool-stuff'] = 10
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'
        REXEE = get_REXEE_instance(params_dict)  # this calss reformat_MDP internally
        assert REXEE.reformatted_mdp is True
        assert REXEE.template['cool_stuff'] == 10
        assert os.path.isfile(os.path.join(input_path, "expanded_test_backup.mdp"))

        os.remove(os.path.join(input_path, "expanded_test.mdp"))
        os.remove(os.path.join(input_path, "expanded_test_backup.mdp"))

    def test_print_params(self, capfd, params_dict):
        # capfd is a fixture in pytest for testing STDOUT
        REXEE = get_REXEE_instance(params_dict)
        REXEE.print_params()
        out, err = capfd.readouterr()
        L = ""
        L += "Important parameters of REXEE\n=============================\n"
        L += f"Python version: {sys.version}\n"
        L += f"GROMACS executable: {REXEE.gmx_path}\n"  # Easier to pass CI. This is easy to catch anyway
        L += f"GROMACS version: {REXEE.gmx_version}\n"  # Easier to pass CI. This is easy to catch anyway
        L += f"ensemble_md version: {ensemble_md.__version__}\n"
        L += "Simulation inputs: ensemble_md/tests/data/sys.gro, ensemble_md/tests/data/sys.top, ensemble_md/tests/data/expanded.mdp\n"  # noqa: E501
        L += "Verbose log file: True\n"
        L += "Proposal scheme: exhaustive\n"
        L += "Whether to perform weight combination: False\n"
        L += "Type of means for weight combination: simple\n"
        L += "Whether to perform histogram correction: False\n"
        L += "Histogram cutoff for weight correction: -1\n"
        L += "Number of replicas: 4\nNumber of iterations: 10\n"
        L += "Length of each replica: 1.0 ps\nFrequency for checkpointing: 100 iterations\n"
        L += "Total number of states: 9\n"
        L += "Additionally defined swappable states: None\n"
        L += "Additional grompp arguments: None\n"
        L += "Additional runtime arguments: None\n"
        L += "External modules for coordinate manipulation: None\n"
        L += "MDP parameters differing across replicas: None\n"
        L += f"Alchemical ranges of each replica in REXEE:\n  - Replica 0: States {[np.int32(0), np.int32(1), np.int32(2), np.int32(3), np.int32(4), np.int32(5)]}\n"  # noqa: E501
        L += f"  - Replica 1: States {[np.int32(1), np.int32(2), np.int32(3), np.int32(4), np.int32(5), np.int32(6)]}\n  - Replica 2: States {[np.int32(2), np.int32(3), np.int32(4), np.int32(5), np.int32(6), np.int32(7)]}\n"  # noqa: E501
        L += f"  - Replica 3: States {[np.int32(3), np.int32(4), np.int32(5), np.int32(6), np.int32(7), np.int32(8)]}\n"  # noqa: E501
        assert out == L

        REXEE.reformatted_mdp = True  # Just to test the case where REXEE.reformatted_mdp is True
        REXEE.print_params(params_analysis=True)
        out, err = capfd.readouterr()
        L += "\nWhether to build Markov state models and perform relevant analysis: False\n"
        L += "Whether to perform free energy calculations: False\n"
        L += "The step to used in subsampling the DHDL data in free energy calculations, if any: 1\n"
        L += "The chosen free energy estimator for free energy calculations, if any: MBAR\n"
        L += "The method for estimating the uncertainty of free energies in free energy calculations, if any: propagate\n"  # noqa: E501
        L += "The number of bootstrap iterations in the boostrapping method, if used: 50\n"
        L += "The random seed to use in bootstrapping, if used: None\n"
        L += "Note that the input MDP file has been reformatted by replacing hypens with underscores. The original mdp file has been renamed as *backup.mdp.\n"  # noqa: E501
        assert out == L

        REXEE.gro = ['ensemble_md/tests/data/sys.gro', 'ensemble_md/tests/data/sys.gro']  # noqa: E501
        REXEE.top = ['ensemble_md/tests/data/sys.top', 'ensemble_md/tests/data/sys.top']
        REXEE.mdp_args = {'ref_p': [1.0, 1.01, 1.02, 1.03], 'ref_t': [298, 300, 302, 303]}
        REXEE.print_params()
        out_3, err = capfd.readouterr()
        line_1 = 'Simulation inputs: ensemble_md/tests/data/sys.gro, ensemble_md/tests/data/sys.gro, ensemble_md/tests/data/sys.top, ensemble_md/tests/data/sys.top, ensemble_md/tests/data/expanded.mdp\n'  # noqa: E501
        line_2 = 'MDP parameters differing across replicas:\n  - ref_p: [1.0, 1.01, 1.02, 1.03]\n  - ref_t: [298, 300, 302, 303]'  # noqa: E501
        assert line_1 in out_3
        assert line_2 in out_3

    def test_initialize_MDP(self, params_dict):
        params_dict['mdp_args'] = {'ref_p': [1.0, 1.01, 1.02, 1.03], 'ref_t': [298, 300, 302, 303]}
        REXEE = get_REXEE_instance(params_dict)
        MDP = REXEE.initialize_MDP(2)  # the third replica
        assert MDP["ref_p"] == 1.02
        assert MDP["nsteps"] == 500
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP["vdw_lambdas"], [0.00, 0.00, 0.00, 0.25, 0.50, 0.75]
                )
            ]
        )
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP["coul_lambdas"], [0.50, 0.75, 1.00, 1.00, 1.00, 1.00]
                )
            ]
        )
        assert all(
            [a == b for a, b in zip(MDP["init_lambda_weights"], [0, 0, 0, 0, 0, 0])]
        )

    def test_get_ref_dist(self, params_dict):
        params_dict['set_ref_dist'] = [True]
        REXEE = get_REXEE_instance(params_dict)
        REXEE.get_ref_dist('ensemble_md/tests/data/pullx.xvg')
        REXEE.ref_dist = [0.428422]

        with pytest.raises(FileNotFoundError):
            REXEE.get_ref_dist()  # the default f"{self.working_dir}/sim_0/iteration_0/pullx.xvg" would not exist

    def test_update_MDP(self, params_dict):
        new_template = "ensemble_md/tests/data/expanded.mdp"
        iter_idx = 3
        states = [2, 5, 7, 4]
        wl_delta = [0.4, 0.32, 0.256, 0.32]
        weights = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [3.48, 2.78, 3.21, 4.56, 8.79, 0.48],
            [8.45, 0.52, 3.69, 2.43, 4.56, 6.73], ]
        counts = [
            [4, 11, 9, 9, 11, 6],
            [9, 8, 8, 11, 7, 7],
            [3, 1, 1, 9, 15, 21],
            [0, 0, 0, 1, 18, 31],
        ]
        params_dict['set_ref_dist'] = [True]

        REXEE = get_REXEE_instance(params_dict)
        REXEE.equil = [-1, 1, 0, -1]  # i.e., the 3rd replica will use fixed weights in the next iteration
        MDP_1 = REXEE.update_MDP(
            new_template, 2, iter_idx, states, wl_delta, weights)  # third replica

        REXEE.get_ref_dist('ensemble_md/tests/data/pullx.xvg')  # so that we can test the pull code
        MDP_2 = REXEE.update_MDP(
            new_template, 3, iter_idx, states, wl_delta, weights, counts)  # fourth replica

        assert MDP_1["tinit"] == MDP_2["tinit"] == 3
        assert MDP_1["nsteps"] == MDP_2["nsteps"] == 500
        assert MDP_1["init_lambda_state"] == 5
        assert MDP_2["init_lambda_state"] == 1
        assert (MDP_1["init_wl_delta"] == MDP_1["wl_scale"] == MDP_1["wl_ratio"] == "")
        assert (MDP_1["lmc_weights_equil"] == MDP_1["weight_equil_wl_delta"] == "")
        assert MDP_2["init_wl_delta"] == 0.32
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP_1["init_lambda_weights"], [3.48, 2.78, 3.21, 4.56, 8.79, 0.48]
                )
            ]
        )
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP_2["init_lambda_weights"], [8.45, 0.52, 3.69, 2.43, 4.56, 6.73]
                )
            ]
        )
        assert MDP_2['init_histogram_counts'] == [0, 0, 0, 1, 18, 31]
        assert MDP_2['pull_coord1_start'] == 'no'
        assert MDP_2['pull_coord1_init'] == 0.428422

    def test_extract_final_dhdl_info(self, params_dict):
        REXEE = get_REXEE_instance(params_dict)
        dhdl_files = [
            os.path.join(input_path, f"dhdl/dhdl_{i}.xvg") for i in range(REXEE.n_sim)
        ]
        states = REXEE.extract_final_dhdl_info(dhdl_files)
        assert states == [5, 2, 2, 8]

    def test_extract_final_log_info(self, params_dict):
        REXEE = get_REXEE_instance(params_dict)
        log_files = [
            os.path.join(input_path, f"log/EXE_{i}.log") for i in range(REXEE.n_sim)]
        wl_delta, weights, counts = REXEE.extract_final_log_info(log_files)
        assert wl_delta == [0.4, 0.5, 0.5, 0.5]
        assert np.allclose(weights, [
            [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
            [0, 1.22635, 2.30707, 2.44120, 4.10308, 6.03106],
            [0, 0.66431, 1.25475, 0.24443, 0.59472, 0.70726],
            [0, 0.09620, 1.59937, -4.31679, -22.89436, -28.08701], ])
        assert counts == [
            [4, 11, 9, 9, 11, 6],
            [9, 8, 8, 11, 7, 7],
            [3, 1, 1, 9, 15, 21],
            [0, 0, 0, 1, 18, 31], ]
        assert REXEE.equil == [-1, -1, -1, -1]

        # Below is a case where one of the replicas (the first replica) got equilibrated
        log_files[0] = os.path.join(input_path, "log/case2_1.log")  # equilibrated weights
        wl_delta, weights, counts = REXEE.extract_final_log_info(log_files)
        assert np.allclose(REXEE.equil, [6.06, -1, -1, -1])
        assert REXEE.equilibrated_weights == [[0.00000, 1.40453, 2.85258, 2.72480, 3.46220, 5.88607], [], [], []]

    def test_get_averaged_weights(self, params_dict):
        REXEE = get_REXEE_instance(params_dict)
        log_files = [
            os.path.join(input_path, f"log/EXE_{i}.log") for i in range(REXEE.n_sim)]
        avg, err = REXEE.get_averaged_weights(log_files)
        assert REXEE.current_wl_delta == [0.4, 0.5, 0.5, 0.5]
        assert REXEE.updating_weights == [
            [
                [0, 3.83101, 4.95736, 5.63808, 6.07220, 6.13408],
                [0, 3.43101, 3.75736, 5.23808, 4.87220, 5.33408],
                [0, 2.63101, 2.95736, 5.23808, 4.47220, 5.73408],
                [0, 1.83101, 2.55736, 4.43808, 4.47220, 6.13408],
                [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
            ],  # the weights of the first replica at 5 different time frames
            [
                [0, 0.72635, 0.80707, 1.44120, 2.10308, 4.03106],
                [0, 0.72635, 1.30707, 1.44120, 2.10308, 4.53106],
                [0, 0.72635, 2.80707, 2.94120, 4.10308, 6.53106],
                [0, 1.72635, 2.30707, 2.44120, 5.10308, 6.53106],
                [0, 1.22635, 2.30707, 2.44120, 4.10308, 6.03106],
            ],  # the weights of the second replica at 5 different time frames
            [
                [0, -0.33569, -0.24525, 2.74443, 4.59472, 7.70726],
                [0, -0.33569, -0.24525, 2.74443, 3.59472, 3.70726],
                [0, -0.33569, -0.24525, 2.74443, 2.09472, 0.20726],
                [0, -0.33569, -0.24525, 1.74443, -0.90528, -0.79274],
                [0, 0.66431, 1.25475, 0.24443, 0.59472, 0.70726]
            ],  # the weights of the third replica at 5 different time frames
            [
                [0, 0.09620, 1.59937, -4.31679, -14.89436, -16.08701],
                [0, 0.09620, 1.59937, -4.31679, -15.89436, -20.08701],
                [0, 0.09620, 1.59937, -4.31679, -18.39436, -22.58701],
                [0, 0.09620, 1.59937, -4.31679, -20.39436, -25.58701],
                [0, 0.09620, 1.59937, -4.31679, -22.89436, -28.08701]
            ]
        ]
        assert np.allclose(avg[0],  [0, 2.55101, 3.35736, 4.83808, 4.8722, 5.89408])
        assert np.allclose(err[0], [0, 1.14542569, 1.0198039, 0.8, 0.69282032, 0.35777088])

    def test_identify_swappable_pairs(self, params_dict):
        REXEE = get_REXEE_instance(params_dict)
        REXEE.state_ranges = [list(range(i, i + 5)) for i in range(REXEE.n_sim)]  # 5 states per replica
        states = [4, 2, 2, 7]   # This would lead to the swappables: [(0, 1), (0, 2), (1, 2)] in the standard case
        dhdl_files = [os.path.join(input_path, f"dhdl/dhdl_{i}.xvg") for i in range(4)]

        # Case 1: Any case that is not neighboring swap has the same definition for the swappable pairs
        swappables, swap_index, states_for_swap = REXEE.identify_swappable_pairs(states, REXEE.state_ranges, dhdl_files)  # noqa: E501
        assert swappables == [(0, 1), (0, 2), (1, 2)]
        assert swap_index == []
        assert states_for_swap == []

        # Case 2: Neighboring exchange
        REXEE.proposal = 'neighboring'
        swappables, swap_index, states_for_swap = REXEE.identify_swappable_pairs(states, REXEE.state_ranges, dhdl_files)  # noqa: E501
        assert swappables == [(0, 1), (1, 2)]
        assert swap_index == []
        assert states_for_swap == []

        # Case 3: Non-neighboring exchange, with add_swappables
        REXEE.proposal = 'exhaustive'
        REXEE.add_swappables = [[3, 7], [4, 7]]
        states = [4, 3, 2, 7]  # Without add_swappables, the swappables would be [(0, 1), (0, 2), (1, 2)]
        swappables, swap_index, states_for_swap = REXEE.identify_swappable_pairs(states, REXEE.state_ranges, dhdl_files)  # noqa: E501
        assert swappables == [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
        assert swap_index == []
        assert states_for_swap == []

        # Case 4: Random Range
        REXEE = get_REXEE_instance(params_dict)
        REXEE.proposal = 'random_range'
        REXEE.state_ranges = [list(range(i, i + 7)) for i in [0, 7, 14, 21]]
        states = [5, 7, 13, 25]
        REXEE.add_swappables = [[6, 7], [13, 14], [20, 21]]
        REXEE.n_tot = 28
        REXEE.s = 7
        REXEE.template['nstdhdl'] = 100
        REXEE.template['nstxout'] = 1000
        REXEE.template['n_sim'] = 4

        dhdl_files = [
            os.path.join(input_path, f"dhdl/random-range/dhdl_{i}.xvg") for i in range(REXEE.n_sim)
        ]
        swappables, swap_index, states_for_swap = REXEE.identify_swappable_pairs(states, REXEE.state_ranges, dhdl_files)  # noqa: E501

        assert swappables == [[0, 1]]
        assert swap_index[0][0] in [14, 15, 16]
        assert swap_index[0][1] in [10, 12, 16, 19]
        assert states_for_swap[0] == [6, 7]

    def test_propose_swap(self, params_dict):
        random.seed(0)
        REXEE = get_REXEE_instance(params_dict)
        swap_1 = REXEE.propose_swap([])
        swap_2 = REXEE.propose_swap([(0, 1), (0, 2), (1, 2)])
        assert swap_1 == []
        assert swap_2 == (1, 2)

    def test_get_swapping_pattern(self, params_dict):
        # weights are obtained from the log files in data/log, where the last states are 5, 2, 2, 8 (global indices)
        # state_ranges are: 0-5, 1-6, ..., 3-8
        dhdl_files = [os.path.join(input_path, f"dhdl/dhdl_{i}.xvg") for i in range(4)]

        # Test 1: Empty swap list, exhaustive proposal
        REXEE = get_REXEE_instance(params_dict)
        REXEE.verbose = False
        states = [0, 6, 7, 8]  # No swappable pairs
        f = copy.deepcopy(dhdl_files)
        pattern, swap_list, swap_index_accept, states_modified = REXEE.get_swapping_pattern(f, states)
        assert REXEE.n_empty_swappable == 1
        assert REXEE.n_swap_attempts == 0
        assert REXEE.n_rejected == 0
        assert pattern == [0, 1, 2, 3]
        assert swap_list == []
        assert swap_index_accept == []
        assert states_modified == states

        # Test 2: Empty swap list, neighboring proposal
        REXEE = get_REXEE_instance(params_dict)
        REXEE.proposal = 'neighboring'  # n_ex will be set to 1 automatically.
        states = [0, 6, 7, 8]  # No swappable pairs
        f = copy.deepcopy(dhdl_files)
        pattern, swap_list, swap_index_accept, states_modified = REXEE.get_swapping_pattern(f, states)
        print(f'2: {swap_index_accept} {states_modified}')
        assert REXEE.n_empty_swappable == 1
        assert REXEE.n_swap_attempts == 0
        assert REXEE.n_rejected == 0
        assert pattern == [0, 1, 2, 3]
        assert swap_list == []
        assert swap_index_accept == []
        assert states_modified == states

        # Test 3: Single swap (proposal = 'single')
        random.seed(0)
        REXEE = get_REXEE_instance(params_dict)
        REXEE.verbose = True
        REXEE.proposal = 'single'  # n_ex will be set to 1 automatically.
        states = [5, 2, 2, 8]  # swappable pairs: [(0, 1), (0, 2), (1, 2)], swap = (1, 2), accept
        f = copy.deepcopy(dhdl_files)
        pattern, swap_list, swap_index_accept, states_modified = REXEE.get_swapping_pattern(f, states)
        assert REXEE.n_swap_attempts == 1
        assert REXEE.n_rejected == 0
        assert pattern == [0, 2, 1, 3]
        assert swap_list == [(1, 2)]
        assert swap_index_accept == [[-1, -1]]
        assert states_modified == states

        # Test 4: Neighboring swap
        random.seed(0)
        REXEE = get_REXEE_instance(params_dict)
        REXEE.proposal = 'neighboring'  # n_ex will be set to 1 automatically.
        states = [5, 2, 2, 8]  # swappable pairs: [(0, 1), (0, 2), (1, 2)], swap = (1, 2), accept
        f = copy.deepcopy(dhdl_files)
        pattern, swap_list, swap_index_accept, states_modified = REXEE.get_swapping_pattern(f, states)
        assert REXEE.n_swap_attempts == 1
        assert REXEE.n_rejected == 0
        assert pattern == [0, 2, 1, 3]
        assert swap_list == [(1, 2)]
        assert swap_index_accept == [[-1, -1]]
        assert states_modified == states

        # Test 5: Exhaustive swaps that end up in a single swap
        random.seed(0)
        REXEE = get_REXEE_instance(params_dict)
        REXEE.proposal = 'exhaustive'
        states = [5, 2, 2, 8]  # swappable pairs: [(0, 1), (0, 2), (1, 2)], swap = (1, 2), accept
        f = copy.deepcopy(dhdl_files)
        pattern, swap_list, swap_index_accept, states_modified = REXEE.get_swapping_pattern(f, states)
        assert REXEE.n_swap_attempts == 1
        assert REXEE.n_rejected == 0
        assert pattern == [0, 2, 1, 3]
        assert swap_list == [(1, 2)]
        assert swap_index_accept == [[-1, -1]]
        assert states_modified == states

        # Test 6: Exhaustive swaps that involve multiple attempted swaps
        random.seed(0)
        REXEE = get_REXEE_instance(params_dict)
        REXEE.proposal = 'exhaustive'
        states = [4, 2, 4, 3]  # swappable pairs: [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]; swap 1: (2, 3), accepted; swap 2: (0, 1), accept  # noqa: E501
        f = copy.deepcopy(dhdl_files)
        pattern, swap_list, swap_index_accept, states_modified = REXEE.get_swapping_pattern(f, states)
        assert REXEE.n_swap_attempts == 2   # \Delta is negative for both swaps -> both accepted
        assert REXEE.n_rejected == 0
        assert pattern == [1, 0, 3, 2]
        assert swap_list == [(2, 3), (0, 1)]
        assert swap_index_accept == [[-1, -1], [-1, -1]]
        assert states_modified == states

        # Test 7: REXEE.proposal is set to exhaustive but there is only one swappable pair anyway.
        random.seed(0)
        REXEE = get_REXEE_instance(params_dict)
        REXEE.proposal = 'exhaustive'
        states = [0, 2, 2, 8]  # swappable pair: [(1, 2)], swap: (1, 2), accept
        f = copy.deepcopy(dhdl_files)
        pattern, swap_list, swap_index_accept, states_modified = REXEE.get_swapping_pattern(f, states)
        assert REXEE.n_swap_attempts == 1
        assert REXEE.n_rejected == 0
        assert pattern == [0, 2, 1, 3]
        assert swap_list == [(1, 2)]
        assert swap_index_accept == [[-1, -1]]
        assert states_modified == states

        # Test 8: modify_coords_fn is not None, so swap_bool is always True
        random.seed(0)
        REXEE = get_REXEE_instance(params_dict)
        REXEE.modify_coords_fn = 'Cool'
        states = [5, 2, 2, 8]  # swappable pairs: [(0, 1), (0, 2), (1, 2)], swap = (1, 2), accept
        f = copy.deepcopy(dhdl_files)
        pattern, swap_list, swap_index_accept, states_modified = REXEE.get_swapping_pattern(f, states)
        assert REXEE.n_swap_attempts == 1
        assert REXEE.n_rejected == 0
        assert pattern == [0, 2, 1, 3]
        assert swap_list == [(1, 2)]
        assert swap_index_accept == [[-1, -1]]
        assert states_modified == states

    def test_calc_prob_acc(self, capfd, params_dict):
        # k = 1.380649e-23; NA = 6.0221408e23; T = 298; kT = k * NA * T / 1000 = 2.4777098766670016
        REXEE = get_REXEE_instance(params_dict)
        # REXEE.state_ranges = [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], ..., [3, 4, 5, 6, 7, 8]]
        states = [5, 2, 2, 8]
        shifts = [0, 1, 2, 3]
        dhdl_files = [os.path.join(input_path, f"dhdl/dhdl_{i}.xvg") for i in range(4)]

        # Test 1
        swap = (0, 1)
        prob_acc = REXEE.calc_prob_acc(swap, dhdl_files, states, shifts)
        out, err = capfd.readouterr()
        # dU = (-9.1366697  + 11.0623788)/2.4777098766670016 ~ 0.7772 kT, so p_acc = 0.45968522728859024
        assert prob_acc == pytest.approx(0.45968522728859024)
        assert 'U^i_n - U^i_m = -3.69 kT, U^j_m - U^j_n = 4.46 kT, Total dU: 0.78 kT' in out

        # Test 2
        swap = (0, 2)
        prob_acc = REXEE.calc_prob_acc(swap, dhdl_files, states, shifts)
        out, err = capfd.readouterr()
        # dU = (-9.1366697 + 4.9963939)/2.4777098766670016 ~ -1.6710 kT, so p_acc = 1
        assert prob_acc == 1

    def test_accept_or_reject(self, params_dict):
        REXEE = get_REXEE_instance(params_dict)
        random.seed(0)
        swap_bool_1 = REXEE.accept_or_reject(0)
        swap_bool_2 = REXEE.accept_or_reject(0.8)  # rand = 0.844
        swap_bool_3 = REXEE.accept_or_reject(0.8)  # rand = 0.758

        assert REXEE.n_swap_attempts == 0  # since we didn't use get_swapping_pattern
        assert REXEE.n_rejected == 2
        assert swap_bool_1 is False
        assert swap_bool_2 is False
        assert swap_bool_3 is True

    def test_weight_correction(self, params_dict):
        REXEE = get_REXEE_instance(params_dict)

        # Case 1: Perform weight correction (N_cutoff reached)
        REXEE.N_cutoff = 5000
        REXEE.verbose = False  # just to increase code coverage
        weights = [[0, 10.304, 20.073, 29.364]]
        counts = [[31415, 45701, 55457, 59557]]
        weights = REXEE.weight_correction(weights, counts)
        assert np.allclose(weights, [
            [
                0,
                10.304 + np.log(31415 / 45701),
                20.073 + np.log(45701 / 55457),
                29.364 + np.log(55457 / 59557),
            ]
        ])  # noqa: E501

        # Case 2: Perform weight correction (N_cutoff not reached by both N_k and N_{k-1})
        REXEE.verbose = True
        weights = [[0, 10.304, 20.073, 29.364]]
        counts = [[3141, 4570, 5545, 5955]]
        weights = REXEE.weight_correction(weights, counts)
        assert np.allclose(weights, [[0, 10.304, 20.073, 29.364 + np.log(5545 / 5955)]])

    def test_combine_weights(self, params_dict):
        """
        Here we just test the combined weights.
        """
        REXEE = get_REXEE_instance(params_dict)
        REXEE.n_tot = 6
        REXEE.n_sub = 4
        REXEE.s = 1
        REXEE.n_sim = 3
        REXEE.state_ranges = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]

        # Test 1: simple means
        weights = [[0, 2.1, 4.0, 3.7], [0, 1.7, 1.2, 2.6], [0, -0.4, 0.9, 1.9]]
        w, g_vec = REXEE.combine_weights(weights)
        assert np.allclose(w, [
            [0, 2.1, 3.9, 3.5],
            [0, 1.8, 1.4, 2.75],
            [0, -0.4, 0.95, 1.95]])
        assert np.allclose(list(g_vec), [0, 2.1, 3.9, 3.5, 4.85, 5.85])

        # Test 2: weighted means
        weights = [[0, 2.1, 4.0, 3.7], [0, 1.7, 1.2, 2.6], [0, -0.4, 0.9, 1.9]]
        errors = [[0, 0.1, 0.15, 0.1], [0, 0.12, 0.1, 0.12], [0, 0.12, 0.15, 0.1]]
        w_2, g_vec_2 = REXEE.combine_weights(weights, errors)
        assert np.allclose(w_2, [
            [0, 2.1, 3.86140725, 3.45417313],
            [0, 1.76140725, 1.35417313, 2.71436889],
            [0, -0.40723412, 0.95296164, 1.95296164]])
        assert np.allclose(list(g_vec_2), [0, 2.1, 3.861407249466951, 3.4541731330165306, 4.814368891580968, 5.814368891580968])  # noqa: E501

        # Test 3: MT-REXEE
        REXEE.n_tot = 27
        REXEE.n_sub = 9
        REXEE.s = 9
        REXEE.n_sim = 3
        REXEE.state_ranges = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                              [9, 10, 11, 12, 13, 14, 15, 16, 17],
                              [18, 19, 20, 21, 22, 23, 24, 25, 26]]
        weights = [[0.0, 0.03813, 0.08744, 0.58124, 1.42376, 1.86745, 3.24031, 3.09609, 3.5702],
                   [0.0, 0.26761, 0.12656, 1.28994, 2.58161, 3.40522, 5.14597, 5.1271, 4.71232],
                   [0.0, -0.24036, -0.21415, 0.90212, 1.88597, 3.5445, 4.62079, 4.48149, 4.84671]]
        w_3, g_vec_3 = REXEE.combine_weights(weights)
        assert np.allclose(list(g_vec_3), [0.0, 0.03813, 0.08744, 0.58124, 1.42376, 1.86745, 3.24031, 3.09609, 3.5702, 0.0, 0.26761, 0.12656, 1.28994, 2.58161, 3.40522, 5.14597, 5.1271, 4.71232, 0.0, -0.24036, -0.21415, 0.90212, 1.88597, 3.5445, 4.62079, 4.48149, 4.84671])  # noqa: E501

    def test_histogram_correction(self, params_dict):
        REXEE = get_REXEE_instance(params_dict)
        REXEE.n_tot = 6
        REXEE.n_sub = 5
        REXEE.s = 1
        REXEE.n_sim = 2
        REXEE.state_ranges = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]
        hist = [[416, 332, 130, 71, 61], [303, 181, 123, 143, 260]]

        hist_modified = REXEE.histogram_correction(hist)
        assert hist_modified == [[416, 332, 161, 98, 98], [332, 161, 98, 98, 178]]

    def test_default_coords_fn(self, params_dict):
        def check_file(true_output, test_output):
            for true_line, test_line in zip(true_output, test_output):
                assert len(true_line) == len(test_line)
                true_sep_line = true_line.split(' ')
                while '' in true_sep_line:
                    true_sep_line.remove('')
                test_sep_line = test_line.split(' ')
                while '' in test_sep_line:
                    test_sep_line.remove('')
                if len(true_sep_line) == 9:
                    assert true_line[0:2] == test_line[0:2]
                    assert np.allclose(np.array(true_line[3:5], dtype=float), np.array(test_line[3:5], dtype=float))
                    assert true_line[6:8] == test_line[6:8]
                else:
                    assert true_line == test_line
        REXEE = get_REXEE_instance(params_dict)
        os.system(f'cp {input_path}/coord_swap/residue_connect.csv .')
        os.system(f'cp {input_path}/coord_swap/atom_name_mapping.csv .')
        os.system(f'cp {input_path}/coord_swap/residue_swap_map.csv .')
        REXEE.default_coords_fn(f'{input_path}/coord_swap/sim_A/confout_backup.gro', f'{input_path}/coord_swap/sim_B/confout_backup.gro', [-1, -1])  # noqa: E501

        true_output_A = open(f'{input_path}/coord_swap/output_A.gro', 'r').readlines()
        test_output_A = open(f'{input_path}/coord_swap/sim_B/confout.gro', 'r').readlines()
        true_output_B = open(f'{input_path}/coord_swap/output_B.gro', 'r').readlines()
        test_output_B = open(f'{input_path}/coord_swap/sim_A/confout.gro', 'r').readlines()

        os.remove('residue_connect.csv')
        os.remove('residue_swap_map.csv')

        check_file(true_output_A, test_output_A)
        check_file(true_output_B, test_output_B)

        os.system(f'cp {input_path}/coord_swap/residue_connect_alt.csv residue_connect.csv')
        os.system(f'cp {input_path}/coord_swap/atom_name_mapping_alt.csv atom_name_mapping.csv')
        os.system(f'cp {input_path}/coord_swap/residue_swap_map_alt.csv residue_swap_map.csv')
        REXEE.default_coords_fn(f'{input_path}/coord_swap/sim_C/confout_backup.gro', f'{input_path}/coord_swap/sim_D/confout_backup.gro', [-1, -1])  # noqa: E501

        true_output_C = open(f'{input_path}/coord_swap/output_C.gro', 'r').readlines()
        test_output_C = open(f'{input_path}/coord_swap/sim_D/confout.gro', 'r').readlines()
        true_output_D = open(f'{input_path}/coord_swap/output_D.gro', 'r').readlines()
        test_output_D = open(f'{input_path}/coord_swap/sim_C/confout.gro', 'r').readlines()

        check_file(true_output_C, test_output_C)
        check_file(true_output_D, test_output_D)
        os.remove('residue_connect.csv')
        os.remove('residue_swap_map.csv')
        os.remove('atom_name_mapping.csv')
        os.remove(f'{input_path}/coord_swap/sim_A/confout.gro')
        os.remove(f'{input_path}/coord_swap/sim_B/confout.gro')
        os.remove(f'{input_path}/coord_swap/sim_C/confout.gro')
        os.remove(f'{input_path}/coord_swap/sim_D/confout.gro')

    def test_process_top(self, params_dict):
        import pandas as pd

        REXEE = get_REXEE_instance(params_dict)
        REXEE.gro = [f'{input_path}/coord_swap/A-B.gro',
                     f'{input_path}/coord_swap/B-C.gro',
                     f'{input_path}/coord_swap/C-D.gro',
                     f'{input_path}/coord_swap/D-E.gro',
                     f'{input_path}/coord_swap/E-F.gro']
        REXEE.resname_list = ['A2B', 'B2C', 'C2D', 'D2E', 'E2F']
        REXEE.swap_rep_pattern = [[[0, 1], [1, 0]], [[1, 1], [2, 0]], [[2, 1], [3, 0]], [[3, 1], [4, 0]]]
        REXEE.top = [f'{input_path}/coord_swap/A-B.top',
                     f'{input_path}/coord_swap/B-C.top',
                     f'{input_path}/coord_swap/C-D.top',
                     f'{input_path}/coord_swap/D-E.top',
                     f'{input_path}/coord_swap/E-F.top']
        REXEE.process_top()
        test_res_connect = pd.read_csv('residue_connect.csv')
        test_swap_map = pd.read_csv('residue_swap_map.csv')
        true_res_connect = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')
        true_swap_map = pd.read_csv(f'{input_path}/coord_swap/residue_swap_map.csv')

        assert true_res_connect.equals(test_res_connect)
        assert true_swap_map.equals(test_swap_map)

        os.remove('atom_name_mapping.csv')
        os.remove('residue_connect.csv')
        os.remove('residue_swap_map.csv')
