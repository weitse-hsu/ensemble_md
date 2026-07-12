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
Unit tests for the module analyze_matrix.py.
"""
import os
import pytest
import numpy as np
from ensemble_md.analysis import analyze_matrix
from ensemble_md.utils.exceptions import ParseError, ParameterError

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


def test_calc_transmtx():
    # Case 1: Expanded ensemble where there is a transition matrix
    A1, B1, C1 = analyze_matrix.calc_transmtx(os.path.join(input_path, 'log/EXE.log'))
    A1_expected = np.array([[0.5, 0.34782609, 0.15000001, 0, 0, 0],
       [0.34782609, 0.18181819, 0.15789473, 0.17647059, 0.10526316, 0.125     ],   # noqa: E128, E202, E203
       [0.15000001, 0.15789473, 0.        , 0.14285715, 0.4375    , 0.07692308],   # noqa: E202, E203
       [0.        , 0.17647059, 0.14285715, 0.16666667, 0.2857143 , 0.18181819],   # noqa: E202, E203
       [0.        , 0.10526316, 0.4375    , 0.2857143 , 0.        , 0.23076923],   # noqa: E202, E203
       [0.        , 0.125     , 0.07692308, 0.18181819, 0.23076923, 0.2       ]])  # noqa: E202, E203

    B1_expected = np.array([[0.5035941 , 0.27816516, 0.15014392, 0.06404528, 0.01056806, 0.00325544],  # noqa: E128, E202, E203, E501
       [0.27816516, 0.1430219 , 0.22873016, 0.12587893, 0.0782651 , 0.07551135],   # noqa: E128, E202, E203
       [0.15014392, 0.22873016, 0.12687601, 0.22781774, 0.19645289, 0.14166102],   # noqa: E202, E203
       [0.06404528, 0.12587893, 0.22781774, 0.0986531 , 0.27133784, 0.18215007],   # noqa: E202, E203
       [0.01056806, 0.0782651 , 0.19645289, 0.27133784, 0.06328789, 0.36931726],   # noqa: E202, E203
       [0.00325544, 0.07551135, 0.14166102, 0.18215007, 0.36931726, 0.2982755 ]])  # noqa: E202, E203

    C1_expected = np.array([[-3.5941000e-03,  6.9660930e-02, -1.4391000e-04, -6.4045280e-02, -1.0568060e-02, -3.2554400e-03],  # noqa: E501, E128
       [ 6.9660930e-02,  3.8796290e-02, -7.0835430e-02,  5.0591660e-02, 2.6998060e-02,  4.9488650e-02],  # noqa: E128, E201, E501
       [-1.4391000e-04, -7.0835430e-02, -1.2687601e-01, -8.4960590e-02, 2.4104711e-01, -6.4737940e-02],
       [-6.4045280e-02,  5.0591660e-02, -8.4960590e-02,  6.8013570e-02, 1.4376460e-02, -3.3188000e-04],
       [-1.0568060e-02,  2.6998060e-02,  2.4104711e-01,  1.4376460e-02, -6.3287890e-02, -1.3854803e-01],
       [-3.2554400e-03,  4.9488650e-02, -6.4737940e-02, -3.3188000e-04, -1.3854803e-01, -9.8275500e-02]])

    np.testing.assert_array_almost_equal(A1, A1_expected)
    np.testing.assert_array_almost_equal(B1, B1_expected)
    np.testing.assert_array_almost_equal(C1, C1_expected)

    # Case 2: Expanded ensemble where there is no transition matrix
    log = os.path.join(input_path, 'log/EXE_0.log')
    with pytest.raises(ParseError, match=f'No transition matrices found in {log}.'):
        A2, B2, C2 = analyze_matrix.calc_transmtx(log)

    # Case 3: Hamiltonian replica exchange
    # Note that the transition matrices shown in the log file of different replicas should all be the same.
    # Here we use log/HREX.log, which is a part of the log file from anthracene HREX.
    A3, B3, C3 = analyze_matrix.calc_transmtx(os.path.join(input_path, 'log/HREX.log'), simulation_type='HREX')
    A3_expected = np.array([[0.7869, 0.2041, 0.0087, 0.0003, 0.0000, 0.0000, 0.0000, 0.0000],  # noqa: E128, E202, E203, E501
       [0.2041, 0.7189, 0.0728, 0.0041, 0.0001, 0.0000, 0.0000, 0.0000],   # noqa: E128, E202, E203
       [0.0087, 0.0728, 0.7862, 0.1251, 0.0071, 0.0001, 0.0000, 0.0000],   # noqa: E202, E203
       [0.0003, 0.0041, 0.1251, 0.7492, 0.1162, 0.0051, 0.0000, 0.0000],   # noqa: E202, E203
       [0.0000, 0.0001, 0.0071, 0.1162, 0.7666, 0.1087, 0.0013, 0.0000],   # noqa: E202, E203
       [0.0000, 0.0000, 0.0001, 0.0051, 0.1087, 0.8100, 0.0689, 0.0073],   # noqa: E202, E203
       [0.0000, 0.0000, 0.0000, 0.0000, 0.0013, 0.0689, 0.6797, 0.2501],   # noqa: E202, E203
       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0073, 0.2501, 0.7426]])  # noqa: E202, E203

    np.testing.assert_array_almost_equal(A3, A3_expected)
    assert B3 is None
    assert C3 is None

    # Case 4: Invalid simulation type
    with pytest.raises(ValueError, match='Invalid simulation type test.'):
        analyze_matrix.calc_transmtx(os.path.join(input_path, 'log/EXE.log'), simulation_type='test')


def test_calc_equil_prob(capfd):
    # Case 1: Right stochastic
    mtx = np.array([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2], [0.1, 0.3, 0.6]])
    np.testing.assert_array_almost_equal(analyze_matrix.calc_equil_prob(mtx), np.array([[13/55], [3/11], [27/55]]))

    # Case 2: Left stochastic
    np.testing.assert_array_almost_equal(analyze_matrix.calc_equil_prob(mtx.T), np.array([[13/55], [3/11], [27/55]]))  # noqa: E501

    # Case 3: Neither left or right stochastic
    mtx = np.random.rand(3, 3)
    prob = analyze_matrix.calc_equil_prob(mtx)
    out, err = capfd.readouterr()
    assert prob is None
    assert 'The input transition matrix is neither right nor left stochastic' in out


def test_calc_t_relax():
    # Case 1: spectral_gap_err is specified
    results = analyze_matrix.calc_t_relax(0.5, 0.1, 0.1)
    assert results[0] == 0.2
    assert results[1] == 0.1 * 0.1 / 0.5 ** 2

    # Case 2: spectral_gap_err is not specified
    results = analyze_matrix.calc_t_relax(0.5, 0.1)
    assert results[0] == 0.2
    assert results[1] is None


def test_calc_spectral_gap(capfd):
    # Case 1 (sanity check): doublly stochastic
    mtx = np.array([[0.5, 0.5], [0.5, 0.5]])
    s, err, vals = analyze_matrix.calc_spectral_gap(mtx, n_bootstrap=5)
    assert vals[0] == 1
    assert np.isclose(s, 1)

    # Case 2: Right stochastic
    mtx = np.array([[0.8, 0.2], [0.3, 0.7]])
    s, err, vals = analyze_matrix.calc_spectral_gap(mtx, n_bootstrap=5)
    assert vals[0] == 1
    assert s == 0.5

    # Case 3: Left stochastic
    s, err, vals = analyze_matrix.calc_spectral_gap(mtx.T, n_bootstrap=5)
    assert vals[0] == 1
    assert s == 0.5

    # Case 4: Neither left or right stochastic
    mtx = np.random.rand(3, 3)
    s = analyze_matrix.calc_spectral_gap(mtx, n_bootstrap=5)  # the output should be None
    out, err = capfd.readouterr()
    assert s is None
    assert 'The input transition matrix is neither right nor left stochastic' in out

    # Case 5: row-stochastic (rows sum to 1) but not a valid probability matrix (has a negative
    # entry), so its largest eigenvalue isn't close to 1. This should raise a ParameterError.
    # (This also regression-tests a previous bug where the check used "is False" on the result of
    # np.isclose(), which is a numpy.bool_ and is never Python's False by identity -- making the
    # check dead code that could never raise, regardless of input.)
    mtx = np.array([[1.5, -0.5], [0.3, 0.7]])
    with pytest.raises(ParameterError, match='is not close to 1'):
        analyze_matrix.calc_spectral_gap(mtx, n_bootstrap=5)


def test_split_transmtx():
    mtx = np.array([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
    sub_mtx = analyze_matrix.split_transmtx(mtx, 2, 2)
    result = [
        np.array([[2/3, 1/3], [1/8, 7/8]]),
        np.array([[7/9, 2/9], [1/8, 7/8]])]
    np.testing.assert_array_almost_equal(sub_mtx[0], result[0])
    np.testing.assert_array_almost_equal(sub_mtx[1], result[1])


def test_plot_matrix():
    """
    We can only check if the figures are generated. Not really able to check how they look like.
    """
    # Case 1: All elements matrix[i][j] are larger than 0.005 and title is not specified
    mtx = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]])

    png_name = 'test.png'
    analyze_matrix.plot_matrix(mtx, png_name)
    assert os.path.isfile(png_name)
    os.remove(png_name)

    # Case 2: At least one of the elements matrix[i][j] is smaller than 0.005 and title is specified
    mtx = np.array([
        [0, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]])

    png_name = 'test.png'
    analyze_matrix.plot_matrix(mtx, png_name, title='cool')
    assert os.path.isfile(png_name)
    os.remove(png_name)
