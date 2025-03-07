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
Unit tests for the module analyze_free_energy.py.
"""
import math
import pytest
import numpy as np
from unittest.mock import patch, call, MagicMock
from ensemble_md.utils import utils
from ensemble_md.analysis import analyze_free_energy


@patch('ensemble_md.analysis.analyze_free_energy.alchemlyb')
@patch('ensemble_md.analysis.analyze_free_energy.subsampling')
@patch('ensemble_md.analysis.analyze_free_energy.extract_u_nk')
@patch('ensemble_md.analysis.analyze_free_energy.extract_dHdl')
@patch('ensemble_md.analysis.analyze_free_energy.detect_equilibration')
@patch('ensemble_md.analysis.analyze_free_energy.subsample_correlated_data')
def test_preprocess_data(mock_corr, mock_equil, mock_extract_dhdl, mock_extract_u_nk, mock_subsampling, mock_alchemlyb, capfd):  # noqa: E501
    mock_data, mock_data_series = MagicMock(), MagicMock()
    mock_alchemlyb.concat.return_value = mock_data
    mock_subsampling.u_nk2series.return_value = mock_data_series
    mock_subsampling._prepare_input.return_value = (mock_data, mock_data_series)
    mock_equil.return_value = (10, 5, 18)  # t, g, Neff_max
    mock_data_series.__len__.return_value = 100   # For one of the print statements

    # Set slicing to return different mock objects based on input
    def slicing_side_effect(*args, **kwargs):
        if args[0] == mock_data:
            return mock_data  # Return mock_data when the data object is sliced
        elif args[0] == mock_data_series:
            return mock_data_series  # Return mock_data_series when the data_series object is sliced

    mock_subsampling.slicing.side_effect = slicing_side_effect

    def generic_list_slicing(key):
        # Function like data_series[t:] internally uses the slice object
        if isinstance(key, slice):
            # Create a new mock to represent the sliced part if needed
            # This allows for separate tracking or specific return values for the slice
            slice_mock = MagicMock()
            return slice_mock

    mock_data.__getitem__.side_effect = generic_list_slicing  # so that we can use mock_data[t:]
    mock_data_series.__getitem__.side_effect = slicing_side_effect  # so that we can use mock_data_series[t:]
    mock_data_series_equil = mock_data_series[10:]  # Mock the equilibrated data series, given t=10

    # Case 1: data_type = u_nk
    files = [[f'ensemble_md/tests/data/dhdl/simulation_example/sim_{i}/iteration_{j}/dhdl.xvg' for j in range(3)] for i in range(4)]  # noqa: E501
    results = analyze_free_energy.preprocess_data(files, 300, 'u_nk')

    out, err = capfd.readouterr()

    assert mock_alchemlyb.concat.call_count == 4
    assert mock_extract_u_nk.call_count == 12
    assert mock_subsampling._prepare_input.call_count == 4
    assert mock_subsampling.slicing.call_count == 8
    assert mock_equil.call_count == 4

    for i in range(4):
        for j in range(3):
            assert mock_extract_u_nk.call_args_list[i * 3 + j] == call(files[i][j], T=300)
        assert mock_subsampling._prepare_input.call_args_list[i] == call(mock_data, mock_data_series, drop_duplicates=True, sort=True)  # noqa: E501
        assert mock_subsampling.slicing.call_args_list[2 * i] == call(mock_data, step=1)
        assert mock_subsampling.slicing.call_args_list[2 * i + 1] == call(mock_data_series, step=1)
        assert 'Subsampling and decorrelating the concatenated u_nk data ...' in out
        assert '  Adopted spacing: 1' in out
        assert '  10.0% of the u_nk data was in the equilibrium region and therfore discarded.' in out  # noqa: E501
        assert '  Statistical inefficiency of u_nk: 5.0' in out
        assert '  Number of effective samples: 18' in out
        assert mock_corr.call_args_list[i] == call(mock_data_series_equil, g=5)

    assert len(results[0]) == 4
    assert results[1] == [10, 10, 10, 10]
    assert results[2] == [5, 5, 5, 5]

    # Case 2: data_type = dHdl
    mock_alchemlyb.concat.reset_mock()
    mock_subsampling._prepare_input.reset_mock()
    mock_subsampling.slicing.reset_mock()
    mock_equil.reset_mock()

    mock_subsampling.dhdl2series.return_value = mock_data_series
    mock_subsampling._prepare_input.return_value = (mock_data, mock_data_series)
    mock_data_series.__len__.return_value = 200
    mock_data_series.values.__len__.return_value = 200

    results = analyze_free_energy.preprocess_data(files, 300, 'dhdl', t=10, g=5)
    out, err = capfd.readouterr()

    for i in range(4):
        for j in range(3):
            assert mock_extract_dhdl.call_args_list[i * 3 + j] == call(files[i][j], T=300)
        assert mock_subsampling._prepare_input.call_args_list[i] == call(mock_data, mock_data_series, drop_duplicates=True, sort=True)  # noqa: E501
        assert mock_subsampling.slicing.call_args_list[2 * i] == call(mock_data, step=1)
        assert mock_subsampling.slicing.call_args_list[2 * i + 1] == call(mock_data_series, step=1)
        assert 'Subsampling and decorrelating the concatenated dhdl data ...' in out
        assert '  Adopted spacing: 1' in out
        assert '  5.0% of the dhdl data was in the equilibrium region and therfore discarded.' in out  # noqa: E501
        assert '  Statistical inefficiency of dhdl: 5.0' in out
        assert '  Number of effective samples: 38' in out
        assert mock_corr.call_args_list[i] == call(mock_data_series_equil, g=5)

    assert len(results[0]) == 4
    assert results[1] == []
    assert results[2] == []

    # Case 3: Invalid data_type
    with pytest.raises(ValueError, match="Invalid data_type. Expected 'u_nk' or 'dhdl'."):
        analyze_free_energy.preprocess_data(files, 300, 'xyz')


@pytest.mark.parametrize("method, expected_estimator", [
    ("MBAR", "MBAR estimator"),
    ("BAR", "BAR estimator"),
    ("TI", "TI estimator"),
])
def test_apply_estimators(method, expected_estimator):
    with patch('ensemble_md.analysis.analyze_free_energy.MBAR') as mock_MBAR, \
         patch('ensemble_md.analysis.analyze_free_energy.BAR') as mock_BAR, \
         patch('ensemble_md.analysis.analyze_free_energy.TI') as mock_TI:

        # Setup mock return values for each estimator
        mock_MBAR_instance = MagicMock()
        mock_MBAR.return_value = mock_MBAR_instance
        mock_MBAR_instance.fit.return_value = "MBAR estimator"  # doesn't matter what the value is

        mock_BAR_instance = MagicMock()
        mock_BAR.return_value = mock_BAR_instance
        mock_BAR_instance.fit.return_value = "BAR estimator"  # doesn't matter what the value is

        mock_TI_instance = MagicMock()
        mock_TI.return_value = mock_TI_instance
        mock_TI_instance.fit.return_value = "TI estimator"  # doesn't matter what the value is

        # Create mock data frames
        mock_data = [MagicMock(name=f"DataFrame {i}") for i in range(3)]

        # Call the function
        results = analyze_free_energy._apply_estimators(mock_data, method)

        # Assertions to check correct estimators are used
        assert len(results) == 3
        assert all(result == expected_estimator for result in results), "Estimator results do not match expected"

        # Ensure fit method is called on each data frame and verify calls
        expected_calls = [call(data) for data in mock_data]
        if method == "MBAR":
            mock_MBAR_instance.fit.assert_has_calls(expected_calls)
        elif method == "BAR":
            mock_BAR_instance.fit.assert_has_calls(expected_calls)
        elif method == "TI":
            mock_TI_instance.fit.assert_has_calls(expected_calls)

        # Test with an invalid method
        with pytest.raises(Exception, match='Specified estimator not available.'):
            analyze_free_energy._apply_estimators(mock_data, "XYZ")


def test_calculate_df_adjacent():
    estimator1 = MagicMock()
    estimator2 = MagicMock()

    # Setup delta_f_ and d_delta_f_ matrices for two estimators
    estimator1.delta_f_ = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    estimator1.d_delta_f_ = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.4, 0.1]])
    estimator2.delta_f_ = np.array([[0, 4, 5], [4, 1, 6], [5, 6, 2]])
    estimator2.d_delta_f_ = np.array([[0.2, 0.3, 0.5], [0.3, 0.2, 0.6], [0.5, 0.6, 0.2]])
    estimators = [estimator1, estimator2]

    df_adjacent, df_err_adjacent = analyze_free_energy._calculate_df_adjacent(estimators)

    assert df_adjacent == [[1, 3], [4, 6]]
    assert df_err_adjacent == [[0.2, 0.4], [0.3, 0.6]]


def test_combine_df_adjacent():
    df_adjacent = [[1, 3], [4, 6]]
    df_err_adjacent = [[0.1, 0.1], [0.1, 0.1]]
    state_ranges = [[0, 1, 2], [1, 2, 3]]

    # Test 1: df_err_adjacent is None (in which case err_type is ignored)
    # Note that this test would lead to two harmless RuntimWarnings due to calculations like np.std([1], ddof=1), which return NaN  # noqa: E501
    results = analyze_free_energy._combine_df_adjacent(df_adjacent, state_ranges, None, "propagate")
    assert results[0] == [1, 3.5, 6]
    assert math.isnan(results[1][0])
    assert results[1][1] == np.std([3, 4], ddof=1)
    assert math.isnan(results[1][2])
    assert results[2] == [False, True, False]

    # Test 2: df_err_adjacent is not None and err_type is "std"
    results = analyze_free_energy._combine_df_adjacent(df_adjacent, state_ranges, df_err_adjacent, "std")
    assert results[0] == [1, 3.5, 6]
    np.testing.assert_array_almost_equal(results[1], [0.1, np.std([3, 4], ddof=1), 0.1])
    assert results[2] == [False, True, False]

    # Test 3: df_err_adjacent is not None and err_type is "propagate"
    df_err_adjacent = [[0.1, 0.1], [0.2, 0.1]]  # make the errs different so that the weighted mean will not be equal to simple mean  # noqa: E501
    results = analyze_free_energy._combine_df_adjacent(df_adjacent, state_ranges, df_err_adjacent, "propagate")
    assert results[0] == [1, utils.weighted_mean([3, 4], [0.1, 0.2])[0], 6]
    assert results[1] == [0.1, utils.weighted_mean([3, 4], [0.1, 0.2])[1], 0.1]
    assert results[2] == [False, True, False]


@patch('ensemble_md.analysis.analyze_free_energy._apply_estimators')
@patch('ensemble_md.analysis.analyze_free_energy._combine_df_adjacent')
@patch('ensemble_md.analysis.analyze_free_energy._calculate_df_adjacent')
def test_calculate_free_energy(mock_calc, mock_combine, mock_apply):
    state_ranges = [[0, 1, 2], [1, 2, 3]]
    mock_data = [MagicMock() for _ in range(4)]
    mock_estimators = [MagicMock() for _ in range(4)]
    mock_apply.return_value = mock_estimators
    mock_calc.return_value = ([[1, 3], [4, 6]], [[0.1, 0.1], [0.1, 0.1]])  # df_adjacent, df_err_adjacent
    mock_combine.return_value = ([1, 3.5, 6], [0.1, np.std([3, 4], ddof=1), 0.1], [False, True, False])
    for data in mock_data:
        data.sample.return_value = data  # Always returns itself, simplifying the bootstrapping logic for testing

    # Test 1: err_method == "propagate"
    results = analyze_free_energy.calculate_free_energy(mock_data, state_ranges, "MBAR", err_method="propagate")
    assert results[0] == [0, 1, 4.5, 10.5]
    assert results[1] == [0, 0.1, np.sqrt(0.1 ** 2 + np.std([3, 4], ddof=1) ** 2), np.sqrt(0.1 ** 2 + np.std([3, 4], ddof=1) ** 2 + 0.1 ** 2)]  # noqa: E501
    assert results[2] == mock_estimators

    # Test 2: err_method == "bootstrap"
    mock_combine.return_value = ([1, 3.5, 6], [0.1, np.std([3, 4], ddof=1), 0.1], [False, True, False])  # we need this since df and df_err are modified by the previous test  # noqa: E501
    results = analyze_free_energy.calculate_free_energy(mock_data, state_ranges, "MBAR", err_method="bootstrap", n_bootstrap=10, seed=0)  # noqa: E501
    # Since all bootstrap iterations are the same, error_bootstrap should be [0, 0, 0], df_err = [0.1, 0, 0]
    assert results[0] == [0, 1, 4.5, 10.5]
    assert results[1] == [0, 0.1, 0.1, np.sqrt(0.1 ** 2 + 0.1 ** 2)]
    assert results[2] == mock_estimators

    # Test 3: Invalid err_method
    with pytest.raises(Exception, match='Specified err_method not available.'):
        analyze_free_energy.calculate_free_energy(mock_data, state_ranges, "MBAR", err_method="XYZ")


def test_calculate_df_rmse():
    # Mock estimators setup
    estimator1 = MagicMock()
    estimator2 = MagicMock()

    # Mock the delta_f_ DataFrame-like attributes
    # Using np.array to simulate the DataFrame iloc functionality
    estimator1.delta_f_ = MagicMock()

    # vals = np.array([0, 1, 2, 3])

    estimator1.delta_f_.iloc.__getitem__.side_effect = lambda x: np.array([0, 1, 2, 3]) if x == 0 else np.array([])  # so estimator1.delta_f_.iloc[0] will be np.array([0, 1, 2, 3])  # noqa: E501
    estimator2.delta_f_ = MagicMock()
    estimator2.delta_f_.iloc.__getitem__.side_effect = lambda x: np.array([0, 1.5, 2.5, 3.5]) if x == 0 else np.array([])  # so estimator1.delta_f_.iloc[0] will be np.array([0, 1.5, 2.5, 3.5])  # noqa: E501
    estimators = [estimator1, estimator2]

    # Reference free energies
    df_ref = [0, 1, 2, 3, 4, 5]
    state_ranges = [[0, 1, 2, 3], [2, 3, 4, 5]]  # Indices into df_ref

    expected_rmse1 = np.sqrt(np.mean((np.array([0, 1, 2, 3]) - np.array([0, 1, 2, 3])) ** 2))
    expected_rmse2 = np.sqrt(np.mean((np.array([0, 1.5, 2.5, 3.5]) - np.array([0, 1, 2, 3])) ** 2))
    expected_rmse = [expected_rmse1, expected_rmse2]

    rmse_list = analyze_free_energy.calculate_df_rmse(estimators, df_ref, state_ranges)
    np.testing.assert_allclose(rmse_list, expected_rmse, rtol=1e-5)


@patch('ensemble_md.analysis.analyze_free_energy.plt')
def test_plot_free_energy(mock_plt):
    analyze_free_energy.plot_free_energy([1, 2, 3], [0.1, 0.1, 0.1], 'test.png')
    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called_once_with(range(3), [1, 2, 3], 'o-', c='#1f77b4')
    mock_plt.errorbar.assert_called_once_with(range(3), [1, 2, 3], yerr=[0.1, 0.1, 0.1], fmt='o', capsize=2, c='#1f77b4')  # noqa: E501
    mock_plt.xlabel.assert_called_once_with('State')
    mock_plt.ylabel.assert_called_once_with('Free energy (kT)')
    mock_plt.grid.assert_called_once()
    mock_plt.savefig.assert_called_once_with('test.png', dpi=600)


def test_average_weights(capfd):
    g_vecs = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    dg_avg, dg_avg_err = analyze_free_energy.average_weights(g_vecs, frac=0.3)
    out, err = capfd.readouterr()
    assert dg_avg == 3
    assert dg_avg_err == 1
    assert 'The number of samples to be averaged is less than 2, so all samples will be averaged.' in out
