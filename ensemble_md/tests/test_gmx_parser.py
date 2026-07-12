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
Unit tests for the module gmx_parser.py. Note that most part of the functions
here have been automatically tested when testing some of the functions in ensemble_md,
so here we are just testing the untested part.
"""
import os
import re
import pytest
import numpy as np
from ensemble_md.utils import gmx_parser
from ensemble_md.utils.exceptions import ParseError

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


def test_parse_mc_lambda_row_no_count_column():
    # A row with fewer than 2 integer-looking tokens (i.e. no row index + Count pair) should
    # raise a clear ParseError instead of silently misidentifying some other column as Count.
    with pytest.raises(ParseError, match="could not identify the 'Count' column"):
        gmx_parser._parse_mc_lambda_row(['0.000', '0.250', '1.234'])


def test_parse_log():
    # Case 1: weight-updating simulation
    weights_0, counts_0, wl_delta_0, equil_time_0 = gmx_parser.parse_log(os.path.join(input_path, 'log/EXE_0.log'))
    assert len(weights_0) == 5
    assert weights_0[0] == [0.0, 3.83101, 4.95736, 5.63808, 6.0722, 6.13408]
    assert weights_0[-1] == [0.00000, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408]
    assert counts_0 == [4, 11, 9, 9, 11, 6]
    assert wl_delta_0 == 0.4
    assert np.isclose(equil_time_0, -1)

    # Case 2-1: equil_time < 1000 ps
    # As a note, here we used a very lenient weight-equil-wl-delta as 0.3, with nst_sim = 2000.
    # The log file was from sim_0/iteration_1.
    weights_1, counts_1, wl_delta_1, equil_time_1 = gmx_parser.parse_log(os.path.join(input_path, 'log/case2_1.log'))
    assert len(weights_1) == 6
    assert weights_1[0] == [0.0, 2.68453, 4.13258, 4.3248, 4.4222, 6.52607]
    assert weights_1[-1] == [0.00000, 1.40453, 2.85258, 2.72480, 3.46220, 5.88607]
    assert counts_1 == [11, 13, 15, 4, 4, 50]
    assert wl_delta_1 is None
    assert np.isclose(equil_time_1, 6.06)

    # Case 2-2: equil_time > 1000 ps (just another case, nothing special here ...)
    # As a note, here we used weight-equil-wl-delta as 0.3, with nst_sim = 2000,  n_iterations = 1000, n_sim = 2,
    # s = 4, wl-scale = 0.9, and wl-ratio = 0.9. The log file was from sim_0/iteration_475 (t=1.9 ns).
    weights_2, counts_2, wl_delta_2, equil_time_2 = gmx_parser.parse_log(os.path.join(input_path, 'log/case2_2.log'))
    assert len(weights_2) == 19
    assert weights_2[0] == [0.00000, 1.60863, 2.47927, 3.15184, 3.47507]
    assert weights_2[-1] == [0.00000, 1.60419, 2.47705, 3.14963, 3.47840]
    assert counts_2 == [2, 3, 1, 2, 1]
    assert wl_delta_2 is None
    assert equil_time_2 == 1903.82

    # Case 3: fixed-weight simulation
    weights_3, counts_3, wl_delta_3, equil_time_3 = gmx_parser.parse_log(os.path.join(input_path, 'log/case3.log'))
    assert len(weights_3) == 1
    assert weights_3[-1] == [0.00000, 1.55165, 2.55043, 3.15034, 3.26889, 4.37831, 5.28574, 3.29638, 2.22527]
    assert counts_3 == [12, 9, 10, 9, 4, 5, 1, 0, 0]
    assert wl_delta_3 is None
    assert equil_time_3 == 0


def test_parse_log_case_insensitive_lmc_stats():
    # lmc-stats = NO (any case) should be recognized as a fixed-weight (Case 3) simulation, not
    # just the lowercase 'no' that the parser happens to be tested against in test_parse_log.
    with open(os.path.join(input_path, 'log/case3.log')) as f:
        content = f.read()
    content_upper = re.sub(r'(lmc-stats\s*=\s*)no\b', r'\1NO', content)
    assert content_upper != content  # sanity check that the substitution actually happened

    with open('case3_upper.log', 'w') as f:
        f.write(content_upper)

    result = gmx_parser.parse_log('case3_upper.log')
    result_ref = gmx_parser.parse_log(os.path.join(input_path, 'log/case3.log'))
    assert result == result_ref

    os.remove('case3_upper.log')


def test_parse_log_missing_n_lambdas():
    # A log file that never prints "n-lambdas" should raise a clear ParseError instead of a
    # cryptic NameError deep inside parsing.
    with open('malformed.log', 'w') as f:
        f.write('This is not a real GROMACS log file.\n')

    with pytest.raises(ParseError, match="could not find the parameter 'n-lambdas'"):
        gmx_parser.parse_log('malformed.log')

    os.remove('malformed.log')


def test_parse_log_missing_weight_updating_params():
    # A log file that has n-lambdas (so it's not immediately rejected) but is missing tinit/dt/
    # weight-equil-wl-delta, and isn't a fixed-weight (lmc-stats = no) log either, should raise a
    # clear ParseError instead of a NameError once the code reaches the Case 1/2 parsing branch.
    with open('malformed.log', 'w') as f:
        f.write('n-lambdas                      = 6\n')

    with pytest.raises(ParseError, match="could not find 'tinit', 'dt', or 'weight-equil-wl-delta'"):
        gmx_parser.parse_log('malformed.log')

    os.remove('malformed.log')


class Test_MDP:
    def test__eq__(self):
        mdp_1 = gmx_parser.MDP("ensemble_md/tests/data/expanded.mdp")
        mdp_2 = gmx_parser.MDP("ensemble_md/tests/data/expanded.mdp")
        assert mdp_1 == mdp_2

    def test_read(self):
        f = open("fake.mdp", "a")
        f.write("TEST")
        f.close()
        with pytest.raises(ParseError, match="'fake.mdp': unknown line in mdp file, 'TEST'"):
            gmx_parser.MDP('fake.mdp')  # This should call the read function in __init__
        os.remove('fake.mdp')

    def test_write(self):
        mdp = gmx_parser.MDP("ensemble_md/tests/data/expanded.mdp")
        mdp.write('test_1.mdp', skipempty=False)
        mdp.write('test_2.mdp', skipempty=True)

        assert os.path.isfile('test_1.mdp')
        assert os.path.isfile('test_2.mdp')

        mdp = gmx_parser.MDP('test_1.mdp')
        mdp.write(skipempty=True)  # This should overwrite the file

        # Check if the files are the same
        with open('test_1.mdp', 'r') as f:
            lines_1 = f.readlines()
        with open('test_2.mdp', 'r') as f:
            lines_2 = f.readlines()
        assert lines_1 == lines_2

        os.remove('test_1.mdp')
        os.remove('test_2.mdp')


def test_compare_MDPs():
    mdp_list = ['ensemble_md/tests/data/mdp/compare_1.mdp', 'ensemble_md/tests/data/mdp/compare_2.mdp', 'ensemble_md/tests/data/mdp/compare_3.mdp']  # noqa: E501
    result_1 = gmx_parser.compare_MDPs(mdp_list[:2], print_diff=True)
    result_2 = gmx_parser.compare_MDPs(mdp_list[1:], print_diff=True)
    dict_1 = {}  # the first two are the same but just in different formats
    dict_2 = {
        'nstdhdl': [100, 10],
        'wl_oneovert': [None, 'yes'],
        'weight_equil_wl_delta': [None, 0.001],
        'init_lambda_weights': [[0.0, 57.88597, 112.71883, 163.84425, 210.48097, 253.80261, 294.79849, 333.90408, 370.82669, 406.02515, 438.53116, 468.53751, 496.24649, 521.58417, 544.57404, 565.26697, 583.7337, 599.60651, 613.43958, 624.70471, 633.95947, 638.29785, 642.44977, 646.33551, 649.91626, 651.54779, 652.93359, 654.13263, 654.94073, 655.13086, 655.07239, 654.66443, 653.68683, 652.32123, 650.72308, 649.2381, 647.94586, 646.599, 645.52063, 643.99133], None],  # noqa: E501
        'wl_ratio': [None, 0.7],
        'lmc_weights_equil': [None, 'wl_delta'],
        'lmc_stats': ['no', 'wang_landau'],
        'wl_scale': [None, 0.8],
        'init_wl_delta': [None, 10],
        'lmc_seed': [None, -1],
        'nstexpanded': [100, 10]
    }

    assert result_1 == dict_1
    assert result_2 == dict_2


def test_read_top():
    top_files = ['A-B.top', 'B-C.top', 'C4-C6.top']
    resname_list = ['A2B', 'B2C', 'I2J']

    for resname, top_file in zip(resname_list, top_files):
        top = gmx_parser.read_top(f'{input_path}/coord_swap/{top_file}', resname)
        assert len(top) > 0

    wrong_resname = ['B2C', 'C2D', 'D2E']
    for resname, top_file in zip(wrong_resname, top_files):
        # The error message should report the full path passed in (not just the basename), so
        # a user can tell exactly which file (out of possibly many same-named files in different
        # directories) failed to parse.
        full_path = f'{input_path}/coord_swap/{top_file}'
        with pytest.raises(Exception, match=f'Residue {resname} can not be found in {re.escape(full_path)}'):
            gmx_parser.read_top(full_path, resname)
