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
Unit tests for the module analyze_traj.py.
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, call, MagicMock
from ensemble_md.analysis import analyze_traj

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


def save_and_exclude(f_input, n_exclude, f_output=None):
    """
    Saves a given file as another file while exlucding the last :code:`n_exclude` lines.

    Parameters
    ----------
    f_input : str
        The input file.
    n_exclude : n
        Number of lines to exclude.
    f_output : str
        The output file. If None, the output file will be the file name of the input
        appendd with "_short".
    """
    if f_output is None:
        f_output = f_input.split('.')[0] + '_short.' + f_input.split('.')[1]

    with open(f_input, 'r') as f:
        lines = f.readlines()[:-n_exclude]

    with open(f_output, 'w') as f:
        f.writelines(lines)


def test_extract_state_traj():
    traj, t = analyze_traj.extract_state_traj(os.path.join(input_path, 'dhdl/dhdl_0.xvg'))
    state_list = [
        0, 0, 3, 1, 4, 4, 5, 4, 5, 5,
        4, 4, 5, 4, 2, 4, 5, 2, 1, 2,
        3, 1, 2, 4, 1, 0, 2, 4, 3, 2,
        1, 3, 3, 4, 2, 3, 1, 1, 0, 1,
        2, 3, 1, 0, 1, 4, 3, 1, 3, 2, 5]
    t_true = [0.02 * i + 3 for i in range(len(state_list))]
    assert traj == state_list
    assert np.allclose(t, t_true)


def test_stitch_time_series():
    folder = os.path.join(input_path, 'dhdl/simulation_example')
    files = [[f'{folder}/sim_{i}/iteration_{j}/dhdl.xvg' for j in range(3)] for i in range(4)]
    rep_trajs = np.array([[0, 0, 1], [1, 1, 0], [2, 2, 2], [3, 3, 3]])
    shifts = [0, 1, 2, 3]

    # Test 1
    trajs = analyze_traj.stitch_time_series(files, rep_trajs, shifts)
    assert trajs[0] == [
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1,
        1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5
    ]
    assert trajs[1] == [
        1, 1, 2, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1,
        2, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 2, 3, 3, 3, 2, 2,
        1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 0, 1
    ]
    assert trajs[2] == [
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 2, 3, 4, 3, 3, 2,
        3, 3, 2, 2, 2, 3, 4, 3, 4, 4, 5, 5, 5, 5, 4, 3, 4, 3, 3, 4, 4
    ]
    assert trajs[3] == [
        3, 3, 3, 3, 3, 3, 3, 5, 4, 4, 5, 4, 4, 5, 4, 5, 5, 5, 4, 5,
        4, 4, 5, 4, 5, 5, 4, 5, 5, 5, 4, 5, 5, 4, 5, 4, 5, 4, 5, 5,
        6, 6, 6, 5, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 5, 6, 6, 6, 7, 6, 7
    ]

    assert os.path.exists('state_trajs.npy')
    os.remove('state_trajs.npy')

    # Test 2: Treat the dhdl files as other types of xvg files
    # Here the time series will be read as is and not shifting is done.
    trajs = analyze_traj.stitch_time_series(files, rep_trajs, dhdl=False, col_idx=1)

    assert trajs[0] == [
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4
    ]
    assert trajs[1] == [
        0, 0, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 0, 1
    ]
    assert trajs[2] == [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 2, 1, 1, 0,
        1, 1, 0, 0, 0, 1, 2, 1, 2, 2, 3, 3, 3, 3, 2, 1, 2, 1, 1, 2, 2
    ]
    assert trajs[3] == [
        0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2,
        1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2,
        3, 3, 3, 2, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 2, 3, 3, 3, 4, 3, 4
    ]

    assert os.path.exists('cv_trajs.npy')
    os.remove('cv_trajs.npy')


def test_stitch_time_series_for_sim():
    folder = os.path.join(input_path, 'dhdl/simulation_example')
    files = [[f'{folder}/sim_{i}/iteration_{j}/dhdl.xvg' for j in range(3)] for i in range(4)]
    shifts = [0, 1, 2, 3]

    # Test 1
    trajs = analyze_traj.stitch_time_series_for_sim(files, shifts)

    trajs[0] == [
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 0, 1
    ]

    trajs[1] == [
        1, 1, 2, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1,
        2, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 2, 3, 3, 3, 2, 2,
        1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 0, 1
    ]

    trajs[2] == [
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 2, 3, 4, 3, 3, 2,
        3, 3, 2, 2, 2, 3, 4, 3, 4, 4, 5, 5, 5, 5, 4, 3, 4, 3, 3, 4, 4
    ]

    trajs[3] == [
        3, 3, 3, 3, 3, 3, 3, 5, 4, 4, 5, 4, 4, 5, 4, 5, 5, 5, 4, 5,
        4, 4, 5, 4, 5, 5, 4, 5, 5, 5, 4, 5, 5, 4, 5, 4, 5, 4, 5, 5,
        6, 6, 6, 5, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 5, 6, 6, 6, 7, 6, 7
    ]

    assert os.path.exists('state_trajs_for_sim.npy')
    os.remove('state_trajs_for_sim.npy')

    # Test 2: The case where dhdl is False
    # Here we again use dhdl.xvg files but use dhdl=False with col_idx=1, which corresponds to the state index
    trajs = analyze_traj.stitch_time_series_for_sim(files, dhdl=False, col_idx=1)

    trajs[0] == [
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 0, 1
    ]

    trajs[1] == [
        1, 1, 2, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1,
        2, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 2, 3, 3, 3, 2, 2,
        1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1, 0, 0, 1, 0, 1, 0, 1
    ]

    trajs[2] == [
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 2, 3, 4, 3, 3, 2,
        3, 3, 2, 2, 2, 3, 4, 3, 4, 4, 5, 5, 5, 5, 4, 3, 4, 3, 3, 4, 4
    ]

    trajs[3] == [
        3, 3, 3, 3, 3, 3, 3, 5, 4, 4, 5, 4, 4, 5, 4, 5, 5, 5, 4, 5,
        4, 4, 5, 4, 5, 5, 4, 5, 5, 5, 4, 5, 5, 4, 5, 4, 5, 4, 5, 5,
        6, 6, 6, 5, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 5, 6, 6, 6, 7, 6, 7
    ]

    assert os.path.exists('state_trajs_for_sim.npy')
    os.remove('state_trajs_for_sim.npy')

    # Test 3: Test for discontinuous time series
    # Here, for sim_2, we exclude the last 5 lines for the dhdl.xvg file in iteration_1 to create a gap
    save_and_exclude(f'{folder}/sim_2/iteration_1/dhdl.xvg', 5)
    os.rename(f'{folder}/sim_2/iteration_1/dhdl.xvg', f'{folder}/sim_2/iteration_1/dhdl_temp.xvg')
    os.rename(f'{folder}/sim_2/iteration_1/dhdl_short.xvg', f'{folder}/sim_2/iteration_1/dhdl.xvg')

    match_str = 'The first frame of iteration 2 in replica 2 is not continuous with the last frame of the previous iteration. '  # noqa: E501
    match_str += f'Please check files {folder}/sim_2/iteration_1/dhdl.xvg and {folder}/sim_2/iteration_2/dhdl.xvg'
    with pytest.raises(ValueError, match=match_str):
        trajs = analyze_traj.stitch_time_series_for_sim(files, shifts)

    # Delete dhdl_short.xvg and rename dhdl_temp.xvg back to dhdl.xvg
    os.remove(f'{folder}/sim_2/iteration_1/dhdl.xvg')
    os.rename(f'{folder}/sim_2/iteration_1/dhdl_temp.xvg', f'{folder}/sim_2/iteration_1/dhdl.xvg')


@patch('ensemble_md.analysis.analyze_traj.utils.run_gmx_cmd')
def test_stitch_xtc_trajs(mock_gmx):
    # Here we mock run_gmx_cmd so we don't need to call GROMACS and don't need example xtc files.
    folder = os.path.join(input_path, 'dhdl/simulation_example')
    files = [[f'{folder}/sim_{i}/iteration_{j}/md.xtc' for j in range(3)] for i in range(4)]
    rep_trajs = np.array([[0, 0, 1], [1, 1, 0], [2, 2, 2], [3, 3, 3]])

    mock_rtn, mock_stdout, mock_stderr = MagicMock(), MagicMock(), MagicMock()
    mock_gmx.return_value = mock_rtn, mock_stdout, mock_stderr

    analyze_traj.stitch_xtc_trajs('gmx', files, rep_trajs)

    args_1 = ['gmx', 'trjcat', '-f', f'{folder}/sim_0/iteration_0/md.xtc', f'{folder}/sim_0/iteration_1/md.xtc', f'{folder}/sim_1/iteration_2/md.xtc', '-o', 'traj_0.xtc']  # noqa: E501
    args_2 = ['gmx', 'trjcat', '-f', f'{folder}/sim_1/iteration_0/md.xtc', f'{folder}/sim_1/iteration_1/md.xtc', f'{folder}/sim_0/iteration_2/md.xtc', '-o', 'traj_1.xtc']  # noqa: E501
    args_3 = ['gmx', 'trjcat', '-f', f'{folder}/sim_2/iteration_0/md.xtc', f'{folder}/sim_2/iteration_1/md.xtc', f'{folder}/sim_2/iteration_2/md.xtc', '-o', 'traj_2.xtc']  # noqa: E501
    args_4 = ['gmx', 'trjcat', '-f', f'{folder}/sim_3/iteration_0/md.xtc', f'{folder}/sim_3/iteration_1/md.xtc', f'{folder}/sim_3/iteration_2/md.xtc', '-o', 'traj_3.xtc']  # noqa: E501

    assert mock_gmx.call_count == 4
    assert mock_gmx.call_args_list[0] == call(args_1)
    assert mock_gmx.call_args_list[1] == call(args_2)
    assert mock_gmx.call_args_list[2] == call(args_3)
    assert mock_gmx.call_args_list[3] == call(args_4)


def test_convert_npy2xvg():
    # Create dummy input data
    trajs = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int)
    dt = 0.1  # ps
    subsampling = 2

    os.chdir('ensemble_md/tests/data')
    analyze_traj.convert_npy2xvg(trajs, dt, subsampling)

    assert os.path.exists('traj_0.xvg')
    assert os.path.exists('traj_1.xvg')

    with open('traj_0.xvg', 'r') as f:
        content = f.readlines()
        assert content[0] == '# This file was created by ensemble_md\n'
        assert content[1] == '# Time (ps) v.s. State index\n'
        assert content[2] == '0.0         0\n'
        assert content[3] == '0.2         2\n'

    with open('traj_1.xvg', 'r') as f:
        content = f.readlines()
        assert content[0] == '# This file was created by ensemble_md\n'
        assert content[1] == '# Time (ps) v.s. State index\n'
        assert content[2] == '0.0         4\n'
        assert content[3] == '0.2         6\n'

    os.remove('traj_0.xvg')
    os.remove('traj_1.xvg')

    trajs = np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]])
    analyze_traj.convert_npy2xvg(trajs, dt, subsampling)

    assert os.path.exists('traj_0.xvg')
    assert os.path.exists('traj_1.xvg')

    with open('traj_0.xvg', 'r') as f:
        content = f.readlines()
        assert content[0] == '# This file was created by ensemble_md\n'
        assert content[1] == '# Time (ps) v.s. CV\n'
        assert content[2] == '0.0      0.000000\n'
        assert content[3] == '0.2      0.200000\n'

    with open('traj_1.xvg', 'r') as f:
        content = f.readlines()
        assert content[0] == '# This file was created by ensemble_md\n'
        assert content[1] == '# Time (ps) v.s. CV\n'
        assert content[2] == '0.0      0.400000\n'
        assert content[3] == '0.2      0.600000\n'

    os.remove('traj_0.xvg')
    os.remove('traj_1.xvg')
    os.chdir('../../../')


def test_traj2transmtx():
    traj = [0, 1, 2, 1, 0, 3]
    N = 4  # matrix size

    # Case 1: normalize=False
    array = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]])
    np.testing.assert_array_equal(analyze_traj.traj2transmtx(traj, N, normalize=False), array)

    # Case 2: normalize=True
    # This test would lead to a harmless RuntimeWarnings due to 0/0 in the last row.
    array = np.array([
        [0, 0.5, 0, 0.5],
        [0.5, 0, 0.5, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]])
    np.testing.assert_array_equal(analyze_traj.traj2transmtx(traj, N, normalize=True), array)


@patch('ensemble_md.analysis.analyze_traj.plt')  # to mock and replace plt (which is matplotlib.pyplot) in analyze_traj
def test_plot_rep_trajs(mock_plt):
    # Not much we can do to test the plot function except to verify if the expected matplotlib functions were called.
    fig_name = 'ensemble_md/tests/data/test.png'
    cmap = mock_plt.cm.ocean

    # Case 1: Short trajs without dt and stride
    trajs = np.array([[0, 1, 1, 0, 2], [1, 0, 1, 2, 0], [2, 0, 1, 0, 2]], dtype=int)
    n_sim = len(trajs)
    colors = [cmap(i) for i in np.arange(n_sim) / n_sim]

    analyze_traj.plot_rep_trajs(trajs, fig_name)

    x_input = np.array([0, 1, 2, 3, 4])
    y_input_1 = np.array([0, 1, 1, 0, 2])
    y_input_2 = np.array([1, 0, 1, 2, 0])
    y_input_3 = np.array([2, 0, 1, 0, 2])

    # Verify that the expected matplotlib functions were called
    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called()
    mock_plt.xlabel.assert_called_with('MC moves')
    mock_plt.ylabel.assert_called_with('Replica')
    mock_plt.grid.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.savefig.assert_called_once_with(fig_name, dpi=600)
    assert mock_plt.plot.call_count == len(trajs)

    # mock_plt.plot.assert_any_call(x_input, y_input_1, color=colors[0], label='Trajectory 0')
    # There is a bug in unittest.mock such that there will be an ValueError upon comparisons of two arrays.
    # Therefore, the line above would fail and we can only use assert_called (as shown above) and compare
    # x_input, y_input_1, etc. with the called arguments.
    # Here mock_plt.plot.call_args_list prints as below. Note that a `call` object can be indexed like a tuple
    # where index 0 contains the positional arguments, and index 1 contains the keyword arguments.
    # [
    #     call(array([0, 1, 2, 3, 4]), array([0, 1, 1, 0, 2]), color=<MagicMock name='plt.cm.ocean()' id='140575569521536'>, label='Trajectory 0', marker='o'),  # noqa: E501
    #     call(array([0, 1, 2, 3, 4]), array([1, 0, 1, 2, 0]), color=<MagicMock name='plt.cm.ocean()' id='140575569521536'>, label='Trajectory 1', marker='o'),  # noqa: E501
    #     call(array([0, 1, 2, 3, 4]), array([2, 0, 1, 0, 2]), color=<MagicMock name='plt.cm.ocean()' id='140575569521536'>, label='Trajectory 2', marker='o')   # noqa: E501
    # ]

    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[0][0], (x_input, y_input_1)))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[1][0], (x_input, y_input_2)))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[2][0], (x_input, y_input_3)))
    assert mock_plt.plot.call_args_list[0][1] == {'color': colors[0], 'label': 'Trajectory 0', 'marker': 'o'}
    assert mock_plt.plot.call_args_list[1][1] == {'color': colors[1], 'label': 'Trajectory 1', 'marker': 'o'}
    assert mock_plt.plot.call_args_list[2][1] == {'color': colors[2], 'label': 'Trajectory 2', 'marker': 'o'}

    # Case 2: Short trajs with dt and stride
    mock_plt.reset_mock()

    dt = 0.2  # ps
    stride = 2
    analyze_traj.plot_rep_trajs(trajs, fig_name, dt, stride)
    x_input = np.array([0, 0.4, 0.8])
    y_input_1 = np.array([0, 1, 2])
    y_input_2 = np.array([1, 1, 0])
    y_input_3 = np.array([2, 1, 2])

    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called()
    mock_plt.xlabel.assert_called_with('Time (ps)')
    mock_plt.ylabel.assert_called_with('Replica')
    mock_plt.grid.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.savefig.assert_called_once_with(fig_name, dpi=600)
    assert mock_plt.plot.call_count == len(trajs)

    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[0][0], (x_input, y_input_1)))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[1][0], (x_input, y_input_2)))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[2][0], (x_input, y_input_3)))
    assert mock_plt.plot.call_args_list[0][1] == {'color': colors[0], 'label': 'Trajectory 0', 'marker': 'o'}
    assert mock_plt.plot.call_args_list[1][1] == {'color': colors[1], 'label': 'Trajectory 1', 'marker': 'o'}
    assert mock_plt.plot.call_args_list[2][1] == {'color': colors[2], 'label': 'Trajectory 2', 'marker': 'o'}

    # Case 3: Long trajs with dt and without stride
    mock_plt.reset_mock()

    trajs = np.random.randint(low=0, high=2, size=(3, 2000000))
    analyze_traj.plot_rep_trajs(trajs, fig_name, dt)
    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called()
    mock_plt.xlabel.assert_called_with('Time (ns)')
    mock_plt.ylabel.assert_called_with('Replica')
    mock_plt.grid.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.savefig.assert_called_once_with(fig_name, dpi=600)
    assert mock_plt.plot.call_count == len(trajs)

    # Here we only check the lengths of x and y inputs
    assert len(mock_plt.plot.call_args_list[0][0][0]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[0][0][1]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[1][0][0]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[1][0][1]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[2][0][0]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[2][0][1]) == 2000000 / 100
    assert mock_plt.plot.call_args_list[0][1] == {'color': colors[0], 'label': 'Trajectory 0'}
    assert mock_plt.plot.call_args_list[1][1] == {'color': colors[1], 'label': 'Trajectory 1'}
    assert mock_plt.plot.call_args_list[2][1] == {'color': colors[2], 'label': 'Trajectory 2'}


@patch('ensemble_md.analysis.analyze_traj.plt')
def test_plot_state_trajs(mock_plt):
    state_ranges = [[0, 1, 2, 3], [2, 3, 4, 5]]
    fig_name = 'ensemble_md/tests/data/test.png'
    cmap = mock_plt.cm.ocean
    n_sim = len(state_ranges)
    colors = [cmap(i) for i in np.arange(n_sim) / n_sim]

    # Mock the return value of plt.subplots to return a tuple of two mock objects
    # We need this because plot_state_trajs calls _, ax = plt.subplots(...). When we mock
    # matplolib.pyplot using mock_plt, plt.subplots will be replaced by mock_plt.subplots
    # and will return a mock object, not the tuple of figure and axes objects that the real plt.subplots returns.
    # This would in turn lead to an ValueError. To avoid this, we need to mock the return values of plt.subplots.
    mock_figure = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_figure, mock_axes)

    # Case 1: Short trajs without dt and stride
    trajs = np.array([[0, 1, 0, 2, 3, 4, 3, 4, 5, 4], [2, 3, 4, 5, 4, 3, 2, 1, 0, 1]], dtype=int)

    analyze_traj.plot_state_trajs(trajs, state_ranges, fig_name)

    x_input = np.arange(10)

    mock_plt.subplots.assert_called_once_with(nrows=1, ncols=2, figsize=(5, 2.5))
    mock_plt.subplot.assert_called()
    mock_plt.plot.assert_called()
    mock_plt.fill_between.assert_called()
    mock_plt.xlabel.assert_called_with('MC moves')
    mock_plt.ylabel.assert_called_with('State')
    mock_plt.xlim.assert_called_with([0, 9])
    mock_plt.ylim.assert_called_with([-0.2, 5.2])
    mock_plt.grid.assert_called()
    mock_plt.tight_layout.assert_called_once()
    mock_plt.savefig.assert_called_once_with(fig_name, dpi=600)

    assert mock_plt.subplot.call_count == len(state_ranges)
    assert mock_plt.plot.call_count == len(state_ranges)
    assert mock_plt.fill_between.call_count == len(state_ranges) ** 2
    assert mock_plt.xlabel.call_count == len(state_ranges)
    assert mock_plt.ylabel.call_count == len(state_ranges)
    assert mock_plt.xlim.call_count == len(state_ranges)
    assert mock_plt.ylim.call_count == len(state_ranges)
    assert mock_plt.grid.call_count == len(state_ranges)

    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[0][0], (x_input, trajs[0])))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[1][0], (x_input, trajs[1])))
    assert mock_plt.plot.call_args_list[0][1] == {'color': colors[0], 'linewidth': 1}
    assert mock_plt.plot.call_args_list[1][1] == {'color': colors[1], 'linewidth': 1}
    assert mock_plt.fill_between.call_args_list[0] == (([0, 9],), {'y1': 3, 'y2': -0.5, 'color': colors[0], 'alpha': 0.1})  # noqa: E501
    assert mock_plt.fill_between.call_args_list[1] == (([0, 9],), {'y1': 5.5, 'y2': 2, 'color': colors[1], 'alpha': 0.1})  # noqa: E501

    # Case 2: Short trajs with dt and stride
    # Well here we will just test things different from Case 1.
    mock_plt.reset_mock()
    dt = 0.2  # ps
    stride = 2
    x_input = np.arange(10)[::stride] * dt
    y_input_1 = np.array([0, 0, 3, 3, 5])
    y_input_2 = np.array([2, 4, 4, 2, 0])

    analyze_traj.plot_state_trajs(trajs, state_ranges, fig_name, dt, stride)

    mock_plt.xlabel.assert_called_with('Time (ps)')
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[0][0], (x_input, y_input_1)))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[1][0], (x_input, y_input_2)))

    # Case 3: Long trajs with dt and without stride
    print('case 3')
    mock_plt.reset_mock()
    trajs = np.random.randint(low=0, high=5, size=(2, 2000000))
    analyze_traj.plot_state_trajs(trajs, state_ranges, fig_name, dt)

    mock_plt.xlabel.assert_called_with('Time (ns)')
    assert len(mock_plt.plot.call_args_list[0][0][0]) == 2000000 / 10
    assert len(mock_plt.plot.call_args_list[0][0][1]) == 2000000 / 10
    assert len(mock_plt.plot.call_args_list[1][0][0]) == 2000000 / 10
    assert len(mock_plt.plot.call_args_list[1][0][1]) == 2000000 / 10
    assert mock_plt.plot.call_args_list[0][1] == {'color': colors[0], 'linewidth': 0.01}
    assert mock_plt.plot.call_args_list[1][1] == {'color': colors[1], 'linewidth': 0.01}


@patch('ensemble_md.analysis.analyze_traj.plt')
def test_plot_state_hist(mock_plt):
    fig_name = 'ensemble_md/tests/data/test.png'
    state_ranges = [[0, 1, 2, 3], [2, 3, 4, 5]]
    trajs = np.array([[0, 1, 0, 2, 3, 4, 3, 4, 5, 4], [2, 3, 4, 5, 4, 3, 2, 1, 0, 1]], dtype=int)
    cmap = mock_plt.cm.ocean
    mock_fig = MagicMock()
    mock_plt.figure.return_value = mock_fig

    n_configs = 2
    colors = [cmap(i) for i in np.arange(n_configs) / n_configs]
    hist_data = np.array([[2, 1, 1, 2, 3, 1], [1, 2, 2, 2, 2, 1]])

    # Case 1: Default settings
    analyze_traj.plot_state_hist(trajs, state_ranges, fig_name)

    mock_plt.figure.assert_called_once_with(figsize=(6.4, 4.8))
    mock_fig.add_subplot.assert_called_once_with(111)
    mock_plt.xticks.assert_called_once_with(range(6))
    mock_plt.xlim.assert_called_once_with([-0.5, 5.5])
    mock_plt.ylim.assert_called_once_with([0, 5.25])  # y_max = (2 + 3) * 1.05
    mock_plt.xlabel.assert_called_once_with('State index')
    mock_plt.ylabel.assert_called_once_with('Count')
    mock_plt.grid.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.tight_layout.assert_called_once()
    mock_plt.savefig.assert_called_once_with(fig_name, dpi=600)

    assert mock_plt.bar.call_count == n_configs
    assert mock_plt.fill_betweenx.call_count == n_configs
    assert mock_plt.fill_betweenx.call_args_list[0] == call([0, 5.25], x1=3.5, x2=-1.0, color=colors[0], alpha=0.1, zorder=0)  # noqa: E501
    assert mock_plt.fill_betweenx.call_args_list[1] == call([0, 5.25], x1=6.0, x2=1.5, color=colors[1], alpha=0.1, zorder=0)  # noqa: E501
    assert mock_plt.bar.call_args_list[0][0][0] == range(6)
    np.testing.assert_array_equal(mock_plt.bar.call_args_list[0][0][1], hist_data[0])
    assert mock_plt.bar.call_args_list[1][0][0] == range(6)
    np.testing.assert_array_equal(mock_plt.bar.call_args_list[1][0][1], hist_data[1])
    assert mock_plt.bar.call_args_list[0][1] == {
        'align': 'center',
        'width': 1,
        'color': colors[0],
        'edgecolor': 'black',
        'label': 'Trajectory 0',
        'alpha': 0.5,
        'bottom': [0, 0, 0, 0, 0, 0]
    }
    assert mock_plt.bar.call_args_list[1][1] == {
        'align': 'center',
        'width': 1,
        'color': colors[1],
        'edgecolor': 'black',
        'label': 'Trajectory 1',
        'alpha': 0.5,
        'bottom': [2, 1, 1, 2, 3, 1]
    }

    # Case 2: max(trajs[-1]) > 30, in which case we can just test the figsize
    trajs_ = np.random.randint(low=29, high=50, size=(2, 200))
    mock_plt.reset_mock()

    analyze_traj.plot_state_hist(trajs_, state_ranges, fig_name)
    mock_plt.figure.assert_called_once_with(figsize=(10, 4.8))

    # Case 3: subplots=True
    mock_plt.reset_mock()
    mock_figure = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_figure, mock_axes)

    analyze_traj.plot_state_hist(trajs, state_ranges, fig_name, subplots=True)

    n_rows, n_cols = 1, 2
    mock_plt.figure.assert_called_once_with(figsize=(6.4, 4.8))
    mock_plt.subplots.assert_called_once_with(nrows=n_rows, ncols=n_cols, figsize=(8, 3))
    mock_plt.xlabel.assert_called_with('State index')
    mock_plt.ylabel.assert_called_with('Count')
    mock_plt.tight_layout.assert_called_once()
    mock_plt.savefig.assert_called_once_with(fig_name, dpi=600)

    assert mock_plt.subplot.call_count == n_configs
    assert mock_plt.subplot.call_args_list[0][0] == (n_rows, n_cols, 1)
    assert mock_plt.subplot.call_args_list[1][0] == (n_rows, n_cols, 2)
    assert mock_plt.bar.call_count == n_configs
    assert mock_plt.xticks.call_count == n_configs
    assert mock_plt.xlim.call_count == n_configs
    assert mock_plt.xlabel.call_count == n_configs
    assert mock_plt.ylabel.call_count == n_configs
    assert mock_plt.title.call_count == n_configs
    assert mock_plt.grid.call_count == n_configs

    assert mock_plt.xticks.call_args_list[0] == call([0, 1, 2, 3], fontsize=8)
    assert mock_plt.xticks.call_args_list[1] == call([2, 3, 4, 5], fontsize=8)
    assert mock_plt.xlim.call_args_list[0] == call([-0.5, 3.5])
    assert mock_plt.xlim.call_args_list[1] == call([1.5, 5.5])
    assert mock_plt.title.call_args_list[0] == call('Trajectory 0')
    assert mock_plt.title.call_args_list[1] == call('Trajectory 1')
    assert mock_plt.bar.call_args_list[0][0][0] == [0, 1, 2, 3]
    assert mock_plt.bar.call_args_list[1][0][0] == [2, 3, 4, 5]
    np.testing.assert_array_equal(mock_plt.bar.call_args_list[0][0][1], hist_data[0][[0, 1, 2, 3]])
    np.testing.assert_array_equal(mock_plt.bar.call_args_list[1][0][1], hist_data[1][[2, 3, 4, 5]])
    assert mock_plt.bar.call_args_list[0][1] == {
        'align': 'center',
        'width': 1,
        'edgecolor': 'black',
        'alpha': 0.5,
    }
    assert mock_plt.bar.call_args_list[1][1] == {
        'align': 'center',
        'width': 1,
        'edgecolor': 'black',
        'alpha': 0.5,
    }

    # Clean up
    os.remove('ensemble_md/tests/data/hist_data.npy')


def test_calc_hist_rmse():
    # Case 1: Exactly flat histogram with some states acceessible by 2 replicas
    hist_data = [[15, 15, 30, 30, 15, 15], [15, 15, 30, 30, 15, 15]]
    state_ranges = [[0, 1, 2, 3], [2, 3, 4, 5]]
    assert analyze_traj.calc_hist_rmse(hist_data, state_ranges) == 0

    # Case 2: Exactly flat histogram with some states acceessible by 3 replicas
    hist_data = [[10, 20, 30, 30, 20, 10], [10, 20, 30, 30, 20, 10]]
    state_ranges = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
    assert analyze_traj.calc_hist_rmse(hist_data, state_ranges) == 0


@patch('ensemble_md.analysis.analyze_traj.plt')
def test_plot_transit_time(mock_plt):
    N = 4
    trajs = [
        [0, 1, 2, 0, 2, 3, 2, 2, 1, 0, 1, 1, 2, 0, 1, 2, 3, 2, 1, 0],
        [1, 2, 1, 0, 1, 2, 2, 3, 2, 3, 3, 2, 1, 0, 1, 2, 2, 3, 2, 1]
    ]

    # Case 1: Default settings
    t_1, t_2, t_3, u = analyze_traj.plot_transit_time(trajs, N)
    assert t_1 == [[5, 7], [4, 4]]
    assert t_2 == [[4, 3], [6]]
    assert t_3 == [[9, 10], [10]]
    assert u == 'step'

    mock_plt.figure.assert_called()
    mock_plt.plot.assert_called()
    mock_plt.xlabel.assert_called_with('Event index')
    mock_plt.ylabel.assert_called()

    assert mock_plt.figure.call_count == 3
    assert mock_plt.plot.call_count == 6
    assert mock_plt.xlabel.call_count == 3
    assert mock_plt.ylabel.call_count == 3
    assert mock_plt.grid.call_count == 3
    assert mock_plt.legend.call_count == 3

    np.testing.assert_array_equal(mock_plt.plot.call_args_list[0][0], [[1, 2], [5, 7]])
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[1][0], [[1, 2], [4, 4]])
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[2][0], [[1, 2], [4, 3]])
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[3][0], [[1], [6]])
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[4][0], [[1, 2], [9, 10]])
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[5][0], [[1], [10]])

    assert [mock_plt.plot.call_args_list[i][1] for i in range(6)] == [
        {'label': 'Trajectory 0', 'marker': 'o'},
        {'label': 'Trajectory 1', 'marker': 'o'},
        {'label': 'Trajectory 0', 'marker': 'o'},
        {'label': 'Trajectory 1', 'marker': 'o'},
        {'label': 'Trajectory 0', 'marker': 'o'},
        {'label': 'Trajectory 1', 'marker': 'o'}
    ]

    assert mock_plt.ylabel.call_args_list[0] == call('Average transit time from states 0 to k (step)')
    assert mock_plt.ylabel.call_args_list[1] == call('Average transit time from states k to 0 (step)')
    assert mock_plt.ylabel.call_args_list[2] == call('Average round-trip time (step)')
    assert [mock_plt.savefig.call_args_list[i][0][0] for i in range(3)] == [
        './t_0k.png',
        './t_k0.png',
        './t_roundtrip.png',
    ]

    # Case 2: dt = 0.2 ps, fig_prefix = 'test', here we just test the return values
    mock_plt.reset_mock()
    t_1, t_2, t_3, u = analyze_traj.plot_transit_time(trajs, N, dt=0.2, fig_prefix='test')
    t_1_, t_2_, t_3_ = [[1.0, 1.4], [0.8, 0.8]], [[0.8, 0.6], [1.2]], [[1.8, 2.0], [2.0]]
    for i in range(2):
        np.testing.assert_array_almost_equal(t_1[i], t_1_[i])
        np.testing.assert_array_almost_equal(t_2[i], t_2_[i])
        np.testing.assert_array_almost_equal(t_3[i], t_3_[i])
    assert u == 'ps'
    assert [mock_plt.savefig.call_args_list[i][0][0] for i in range(3)] == [
        './test_t_0k.png',
        './test_t_k0.png',
        './test_t_roundtrip.png',
    ]

    # Case 3: dt = 200 ps, long trajs
    mock_plt.reset_mock()
    trajs = np.ones((2, 2000000), dtype=int)
    trajs[0][0], trajs[0][1000000], trajs[0][1999999] = 0, 3, 0
    t_1, t_2, t_3, u = analyze_traj.plot_transit_time(trajs, N, dt=200)
    assert t_1 == [[200000.0], []]
    assert t_2 == [[199999.8], []]
    assert t_3 == [[399999.8], []]
    assert u == 'ns'
    mock_plt.ticklabel_format.assert_called_with(style='sci', axis='y', scilimits=(0, 0))
    assert mock_plt.ticklabel_format.call_count == 3

    # Case 4: Poor sampling
    mock_plt.reset_mock()
    trajs = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]
    t_1, t_2, t_3, u = analyze_traj.plot_transit_time(trajs, N)
    assert t_1 == [[], []]
    assert t_2 == [[], []]
    assert t_3 == [[], []]
    assert u == 'step'
    mock_plt.figure.assert_not_called()
    mock_plt.savefig.assert_not_called()

    # Case 5: More than 100 round trips so that a histogram is plotted
    mock_plt.reset_mock()
    trajs = np.array([[0, 1, 2, 3, 2] * 20000, [0, 1, 3, 2, 1] * 20000])
    t_1, t_2, t_3, u = analyze_traj.plot_transit_time(trajs, N, fig_prefix='test')

    assert t_1 == [[3] * 20000, [2] * 20000]
    assert t_2 == [[2] * 19999, [3] * 19999]
    assert t_3 == [[5] * 19999, [5] * 19999]
    assert u == 'step'

    mock_plt.hist.assert_called()
    mock_plt.ticklabel_format.assert_called_with(style='sci', axis='y', scilimits=(0, 0))

    assert mock_plt.figure.call_count == 6
    assert mock_plt.hist.call_count == 6
    assert mock_plt.xlabel.call_count == 6
    assert mock_plt.ylabel.call_count == 6
    assert mock_plt.ticklabel_format.call_count == 6
    assert mock_plt.grid.call_count == 6
    assert mock_plt.legend.call_count == 6
    assert mock_plt.savefig.call_count == 6

    assert mock_plt.hist.call_args_list[0][0][0] == [3] * 20000
    assert mock_plt.hist.call_args_list[1][0][0] == [2] * 20000
    assert mock_plt.hist.call_args_list[2][0][0] == [2] * 19999
    assert mock_plt.hist.call_args_list[3][0][0] == [3] * 19999
    assert mock_plt.hist.call_args_list[4][0][0] == [5] * 19999
    assert mock_plt.hist.call_args_list[5][0][0] == [5] * 19999

    assert [mock_plt.hist.call_args_list[i][1] for i in range(6)] == [
        {'bins': 1000, 'label': 'Trajectory 0'},
        {'bins': 1000, 'label': 'Trajectory 1'},
        {'bins': 999, 'label': 'Trajectory 0'},
        {'bins': 999, 'label': 'Trajectory 1'},
        {'bins': 999, 'label': 'Trajectory 0'},
        {'bins': 999, 'label': 'Trajectory 1'}
    ]

    assert [mock_plt.xlabel.call_args_list[i][0][0] for i in range(6)] == [
        'Event index',
        'Average transit time from states 0 to k (step)',
        'Event index',
        'Average transit time from states k to 0 (step)',
        'Event index',
        'Average round-trip time (step)'
    ]

    assert [mock_plt.ylabel.call_args_list[i][0][0] for i in range(6)] == [
        'Average transit time from states 0 to k (step)',
        'Event count',
        'Average transit time from states k to 0 (step)',
        'Event count',
        'Average round-trip time (step)',
        'Event count'
    ]

    assert [mock_plt.savefig.call_args_list[i][0][0] for i in range(6)] == [
        './test_t_0k.png',
        './test_hist_t_0k.png',
        './test_t_k0.png',
        './test_hist_t_k0.png',
        './test_t_roundtrip.png',
        './test_hist_t_roundtrip.png'
    ]


@patch('ensemble_md.analysis.analyze_traj.plt')
def test_plot_g_vecs(mock_plt):
    cmap = mock_plt.cm.ocean
    mock_ax = MagicMock()
    mock_plt.gca.return_value = mock_ax
    colors = [cmap(i) for i in np.arange(4) / 4]

    # Case 1: Short g_vecs with refs and with plot_rmse = True
    g_vecs = np.array([[0, 10, 20, 30], [0, 8, 18, 28]])
    refs = np.array([0, 8, 18, 28])
    refs_err = np.array([0.1, 0.1, 0.1, 0.1])

    analyze_traj.plot_g_vecs(g_vecs, refs, refs_err, plot_rmse=True)

    mock_plt.figure.assert_called()
    mock_plt.plot.assert_called()
    mock_plt.xlabel.assert_called_with('Iteration index')
    mock_plt.xlim.assert_called()
    mock_plt.grid.assert_called()
    mock_plt.legend.assert_called_once_with(loc='center left', bbox_to_anchor=(1, 0.2))
    mock_plt.xlabel.assert_called_with('Iteration index')

    assert mock_plt.figure.call_count == 2
    assert mock_plt.plot.call_count == 4
    assert mock_plt.axhline.call_count == 3
    assert mock_plt.fill_between.call_count == 3
    assert mock_plt.grid.call_count == 2
    assert mock_plt.xlabel.call_count == 2
    assert mock_plt.ylabel.call_count == 2

    assert [mock_plt.plot.call_args_list[i][0][0] for i in range(4)] == [range(2)] * 4
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[0][0][1], np.array([10, 8]))
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[1][0][1], np.array([20, 18]))
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[2][0][1], np.array([30, 28]))
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[3][0][1], np.array([np.sqrt(3), 0]))  # RMSE as a function the iteration index  # noqa: E501

    assert mock_plt.plot.call_args_list[0][1] == {'label': 'State 1', 'marker': 'o', 'c': colors[0], 'linewidth': 0.8, 'markersize': 2}  # noqa: E501
    assert mock_plt.plot.call_args_list[1][1] == {'label': 'State 2', 'marker': 'o', 'c': colors[1], 'linewidth': 0.8, 'markersize': 2}  # noqa: E501
    assert mock_plt.plot.call_args_list[2][1] == {'label': 'State 3', 'marker': 'o', 'c': colors[2], 'linewidth': 0.8, 'markersize': 2}  # noqa: E501

    assert mock_plt.ylabel.call_args_list[0][0] == ('Alchemical weight (kT)',)
    assert mock_plt.ylabel.call_args_list[1][0] == ('RMSE in the alchemical weights (kT)',)

    # Case 2: Long g_vecs, here we just check the only different line
    mock_plt.reset_mock()
    g_vecs = np.array([[0, 10, 20, 30]] * 200)
    analyze_traj.plot_g_vecs(g_vecs)

    assert mock_plt.plot.call_count == 3
    assert [mock_plt.plot.call_args_list[i][0][0] for i in range(3)] == [range(200)] * 3

    np.testing.assert_array_equal(mock_plt.plot.call_args_list[0][0][1], np.array([10] * 200))
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[1][0][1], np.array([20] * 200))
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[2][0][1], np.array([30] * 200))

    assert mock_plt.plot.call_args_list[0][1] == {'label': 'State 1', 'c': colors[0], 'linewidth': 0.8}
    assert mock_plt.plot.call_args_list[1][1] == {'label': 'State 2', 'c': colors[1], 'linewidth': 0.8}
    assert mock_plt.plot.call_args_list[2][1] == {'label': 'State 3', 'c': colors[2], 'linewidth': 0.8}


def test_get_swaps():
    input_file = os.path.join('ensemble_md/tests/data', 'run_REXEE_log.txt')
    proposed_swaps, accepted_swaps = analyze_traj.get_swaps(input_file)

    assert proposed_swaps == [
        {0: 0, 1: 3, 2: 1, 3: 0, 4: 0},
        {1: 2, 2: 2, 3: 0, 4: 1, 5: 1},
        {2: 3, 3: 3, 4: 2, 5: 0, 6: 0},
        {3: 0, 4: 1, 5: 0, 6: 3, 7: 0},
    ]

    assert accepted_swaps == [
        {0: 0, 1: 2, 2: 1, 3: 0, 4: 0},
        {1: 2, 2: 1, 3: 0, 4: 0, 5: 0},
        {2: 2, 3: 0, 4: 0, 5: 0, 6: 0},
        {3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
    ]


@patch('ensemble_md.analysis.analyze_traj.plt')
def test_plot_swaps(mock_plt):
    swaps = [
        {0: 0, 1: 3, 2: 1, 3: 0, 4: 0},
        {1: 2, 2: 2, 3: 0, 4: 1, 5: 1},
        {2: 3, 3: 3, 4: 2, 5: 0, 6: 0},
        {3: 0, 4: 1, 5: 0, 6: 3, 7: 0},
    ]

    # Test 1: The case not specifying the swap_type
    cmap = mock_plt.cm.ocean
    colors = [cmap(i) for i in np.arange(8) / 8]
    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_plt.figure.return_value = mock_fig
    mock_fig.add_subplot.return_value = mock_ax

    mock_min, mock_max = MagicMock(), MagicMock()
    mock_ax.get_ylim.return_value = (mock_min, mock_max)

    analyze_traj.plot_swaps(swaps, stack=True)

    mock_plt.figure.assert_called_once_with(figsize=(6.4, 4.8))
    mock_fig.add_subplot.assert_called_once_with(111)
    mock_plt.bar.assert_called()
    mock_plt.xticks.assert_called_once_with(range(8))
    mock_plt.fill_betweenx.assert_called()
    mock_plt.xlim.assert_called_once_with([-0.5, 7.5])
    mock_plt.xlabel.assert_called_once_with('State')
    mock_plt.ylabel.assert_called_once_with('Number of swaps')
    mock_plt.grid.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.tight_layout.assert_called_once()
    mock_plt.savefig.assert_called_once_with('swaps.png', dpi=600)

    counts_list = [
        [0, 3, 1, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 1, 1, 0, 0],
        [0, 0, 3, 3, 2, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 3, 0],
    ]
    assert mock_plt.bar.call_count == 4
    assert [mock_plt.bar.call_args_list[i][0][0] for i in range(4)] == [range(8)] * 4
    assert [mock_plt.bar.call_args_list[i][0][1] for i in range(4)] == [counts_list[i] for i in range(4)]
    assert mock_plt.bar.call_args_list[0][1] == {
        'align': 'center',
        'width': 1,
        'color': colors[0],
        'edgecolor': 'black',
        'label': 'Replica 0',
        'alpha': 0.5,
        'bottom': [0, 0, 0, 0, 0, 0, 0, 0]
    }
    assert mock_plt.bar.call_args_list[1][1] == {
        'align': 'center',
        'width': 1,
        'color': colors[1],
        'edgecolor': 'black',
        'label': 'Replica 1',
        'alpha': 0.5,
        'bottom': [0, 3, 1, 0, 0, 0, 0, 0]
    }
    assert mock_plt.bar.call_args_list[2][1] == {
        'align': 'center',
        'width': 1,
        'color': colors[2],
        'edgecolor': 'black',
        'label': 'Replica 2',
        'alpha': 0.5,
        'bottom': [0, 5, 3, 0, 1, 1, 0, 0]
    }
    assert mock_plt.bar.call_args_list[3][1] == {
        'align': 'center',
        'width': 1,
        'color': colors[3],
        'edgecolor': 'black',
        'label': 'Replica 3',
        'alpha': 0.5,
        'bottom': [0, 5, 6, 3, 3, 1, 0, 0]
    }

    # Below we only check the keyword arguments of the fill_betweenx calls
    assert mock_plt.fill_betweenx.call_args_list[0] == call([mock_min, mock_max], x1=4.5, x2=-1, color=colors[0], alpha=0.1, zorder=0)  # noqa: E501
    assert mock_plt.fill_betweenx.call_args_list[1] == call([mock_min, mock_max], x1=5.5, x2=0.5, color=colors[1], alpha=0.1, zorder=0)  # noqa: E501
    assert mock_plt.fill_betweenx.call_args_list[2] == call([mock_min, mock_max], x1=6.5, x2=1.5, color=colors[2], alpha=0.1, zorder=0)  # noqa: E501
    assert mock_plt.fill_betweenx.call_args_list[3] == call([mock_min, mock_max], x1=8, x2=2.5, color=colors[3], alpha=0.1, zorder=0)  # noqa: E501

    # Test 2: The case specifying the swap_type
    mock_plt.reset_mock()
    analyze_traj.plot_swaps(swaps, swap_type='proposed')
    mock_plt.ylabel.assert_called_with('Number of proposed swaps')
    mock_plt.savefig.assert_called_once_with('proposed_swaps.png', dpi=600)


@patch('ensemble_md.analysis.analyze_traj.np')
def test_get_g_evolution(mock_np):
    # Here instead of checking the values of g_vecs_avg and g_vecs_err, we check the inputs to np.mean and np.std

    # Test 1: A standard case where two log files are passed with default parameters
    g_vecs_all, g_vecs_avg, g_vecs_err = analyze_traj.get_g_evolution(
        log_files=['ensemble_md/tests/data/log/EXE_0.log', 'ensemble_md/tests/data/log/EXE_1.log'],
        start_state=0,
        end_state=6
    )
    assert g_vecs_all == [
        [0, 3.83101, 4.95736, 5.63808, 6.07220, 6.13408],
        [0, 3.43101, 3.75736, 5.23808, 4.87220, 5.33408],
        [0, 2.63101, 2.95736, 5.23808, 4.47220, 5.73408],
        [0, 1.83101, 2.55736, 4.43808, 4.47220, 6.13408],
        [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
        [0, 0.72635, 0.80707, 1.44120, 2.10308, 4.03106],
        [0, 0.72635, 1.30707, 1.44120, 2.10308, 4.53106],
        [0, 0.72635, 2.80707, 2.94120, 4.10308, 6.53106],
        [0, 1.72635, 2.30707, 2.44120, 5.10308, 6.53106],
        [0, 1.22635, 2.30707, 2.44120, 4.10308, 6.03106],
    ]
    assert g_vecs_avg is None
    assert g_vecs_err is None

    mock_np.mean.assert_not_called()
    mock_np.std.assert_not_called()

    # Test 2: Test the avg_frac parameter
    mock_np.reset_mock()
    _ = analyze_traj.get_g_evolution(
        log_files=['ensemble_md/tests/data/log/EXE_0.log'],
        start_state=0,
        end_state=6,
        avg_frac=0.5  # the last 2 out of 5 frames should be used
    )

    input_weights = np.array([[0, 1.83101, 2.55736, 4.43808, 4.47220, 6.13408], [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408]])  # noqa: E501

    assert mock_np.mean.call_count == 1
    assert mock_np.std.call_count == 1
    np.testing.assert_array_equal(mock_np.mean.call_args_list[0][0][0], input_weights)
    np.testing.assert_array_equal(mock_np.std.call_args_list[0][0][0], input_weights)
    assert mock_np.mean.call_args_list[0][1] == {'axis': 0}
    assert mock_np.std.call_args_list[0][1] == {'axis': 0, 'ddof': 1}

    # Test 3: Test avg_from_last_update but with a log file where wl-delta was not updated
    mock_np.reset_mock()
    _ = analyze_traj.get_g_evolution(
        ['ensemble_md/tests/data/log/EXE_0.log'],
        start_state=0,
        end_state=6,
        avg_frac=0.5,  # here we check if this option is ignored
        avg_from_last_update=True  # wl-delta was not updated in EXE_0.log so all weights will be used for mean/std
    )

    input_weights = [
        [0, 3.83101, 4.95736, 5.63808, 6.07220, 6.13408],
        [0, 3.43101, 3.75736, 5.23808, 4.87220, 5.33408],
        [0, 2.63101, 2.95736, 5.23808, 4.47220, 5.73408],
        [0, 1.83101, 2.55736, 4.43808, 4.47220, 6.13408],
        [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
    ]

    assert mock_np.mean.call_count == 1
    assert mock_np.std.call_count == 1
    np.testing.assert_array_equal(mock_np.mean.call_args_list[0][0][0], input_weights)
    np.testing.assert_array_equal(mock_np.std.call_args_list[0][0][0], input_weights)
    assert mock_np.mean.call_args_list[0][1] == {'axis': 0}
    assert mock_np.std.call_args_list[0][1] == {'axis': 0, 'ddof': 1}

    # Test 4: Test avg_from_last_update and with a log file where wl-delta was indeed updated and
    # the weights got equilibrated
    mock_np.reset_mock()
    g_vecs_all, g_vecs_avg, g_vecs_err = analyze_traj.get_g_evolution(
        ['ensemble_md/tests/data/log/case2_1.log'],
        start_state=0,
        end_state=6,
        avg_from_last_update=True
    )

    assert g_vecs_all == [
        [0, 1.16453, 2.69258, 2.48480, 1.46220, 3.88607],  # 4.2 ps
        [0, 1.16453, 1.49258, 2.48480, 1.06220, 3.88607],  # 4.4 ps
        [0, 1.16453, 1.89258, 2.08480, 1.86220, 3.88607],  # 4.5 ps
        [0, 1.16453, 1.89258, 2.08480, 1.86220, 4.68607],  # 4.8 ps
        [0, 2.36453, 3.09258, 3.28480, 3.06220, 5.48607],  # 5.0 ps
        [0, 2.68453, 4.13258, 4.32480, 4.42220, 6.52607],  # 5.2 ps, wl-delta updated
        [0, 2.36453, 3.49258, 4.00480, 4.74220, 6.20607],  # 5.4 ps
        [0, 2.36453, 3.17258, 3.36480, 3.78220, 4.92607],  # 5.6 ps
        [0, 1.40453, 2.85258, 2.08480, 3.14220, 4.92607],  # 5.8 ps
        [0, 1.40453, 2.53258, 2.40480, 3.14220, 5.56607],  # 6.0 ps, equilibrated at 6.04 ps
        [0, 1.40453, 2.85258, 2.72480, 3.46220, 5.88607],  # 6.2 ps
    ]

    assert mock_np.mean.call_count == 1
    assert mock_np.std.call_count == 1
    np.testing.assert_array_equal(mock_np.mean.call_args_list[0][0][0], g_vecs_all[-6:])
    np.testing.assert_array_equal(mock_np.std.call_args_list[0][0][0], g_vecs_all[-6:])
    assert mock_np.mean.call_args_list[0][1] == {'axis': 0}
    assert mock_np.std.call_args_list[0][1] == {'axis': 0, 'ddof': 1}


@patch('ensemble_md.analysis.analyze_traj.get_g_evolution')
def test_get_dg_evoluation(mock_fn):
    mock_fn.return_value = ([
        [0, 3.83101, 4.95736, 5.63808, 6.07220, 6.13408],
        [0, 3.43101, 3.75736, 5.23808, 4.87220, 5.33408],
        [0, 2.63101, 2.95736, 5.23808, 4.47220, 5.73408],
        [0, 1.83101, 2.55736, 4.43808, 4.47220, 6.13408],
        [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
    ], MagicMock(), MagicMock())

    # Test 1
    dg = analyze_traj.get_dg_evolution(['ensemble_md/tests/data/log/EXE_0.log'], 0, 3)
    mock_fn.assert_called_once_with(['ensemble_md/tests/data/log/EXE_0.log'], 0, 3)
    np.testing.assert_array_equal(dg, np.array([5.63808, 5.23808, 5.23808, 4.43808, 3.63808]))

    # Test 2 (just different start_state/end_state values)
    mock_fn.reset_mock()
    dg = analyze_traj.get_dg_evolution(['ensemble_md/tests/data/log/EXE_0.log'], 1, 3)
    mock_fn.assert_called_once_with(['ensemble_md/tests/data/log/EXE_0.log'], 1, 3)
    np.testing.assert_array_almost_equal(dg, np.array([1.80707, 1.80707, 2.60707, 2.60707, 2.60707]))


@patch('ensemble_md.analysis.analyze_traj.plt')
@patch('ensemble_md.analysis.analyze_traj.get_dg_evolution')
def test_plot_dg_evolution(mock_fn, mock_plt):  # the outer decorator mock_plt should be the second parameter
    # Test 1: Short dg
    mock_fn.return_value = np.arange(10)
    dg = analyze_traj.plot_dg_evolution(['log_0.log'], 1, 3)  # the values of log_files does not matter since the mocked value of dg is specified anyway  # noqa: E501
    mock_fn.assert_called_once_with(['log_0.log'], 1, 3)
    np.testing.assert_array_equal(dg, np.arange(10))
    t = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

    mock_plt.figure.assert_called()
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[0][0][0], t)
    np.testing.assert_array_equal(mock_plt.plot.call_args_list[0][0][1], dg)
    mock_plt.xlabel.assert_called_once_with('Time (ps)')
    mock_plt.ylabel.assert_called_once_with(r'$\Delta g$')
    mock_plt.grid.assert_called_once()
    mock_plt.savefig.assert_called_once_with('dg_evolution.png', dpi=600)

    # Test 2: Long dg
    mock_fn.reset_mock()
    mock_plt.reset_mock()
    mock_fn.return_value = np.arange(20000)
    dg = analyze_traj.plot_dg_evolution(['log_0.log'], 1, 3, start_idx=100)
    mock_fn.assert_called_once_with(['log_0.log'], 1, 3)
    np.testing.assert_array_equal(dg, np.arange(20000)[100:])
    mock_plt.xlabel.assert_called_once_with('Time (ns)')


@patch('ensemble_md.analysis.analyze_traj.plt')
def test_get_delta_w_updates(mock_plt):
    # Test 1
    t_updates, delta_w_updates, equil = analyze_traj.get_delta_w_updates('ensemble_md/tests/data/log/case2_1.log', plot=True)  # noqa: E501
    np.testing.assert_almost_equal(t_updates, [0, 0.00104, 0.00204])
    assert delta_w_updates == [0.4, 0.32, 0.256]
    assert equil is True

    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called()
    mock_plt.xlabel.assert_called_once_with('Time (ns)')
    mock_plt.ylabel.assert_called_once_with(r'Wang-Landau incrementor ($k_{B}T$)')
    mock_plt.grid.cassert_called_once()
    mock_plt.savefig.assert_called_once_with('delta_w_updates.png', dpi=600)

    assert mock_plt.plot.call_count == 4
    assert mock_plt.text.call_count == 3
    assert mock_plt.plot.call_args_list[0][0][0] == t_updates[:2]
    assert mock_plt.plot.call_args_list[0][0][1] == [0.4, 0.4]
    assert mock_plt.plot.call_args_list[1][0][0] == [t_updates[1], t_updates[1]]
    assert mock_plt.plot.call_args_list[1][0][1] == [0.4, 0.32]
    assert mock_plt.plot.call_args_list[2][0][0] == t_updates[1:]
    assert mock_plt.plot.call_args_list[2][0][1] == [0.32, 0.32]
    assert mock_plt.plot.call_args_list[3][0][0] == [0.00204, 0.00204]
    assert mock_plt.plot.call_args_list[3][0][1] == [0.32, 0.256]
    assert mock_plt.text.call_args_list[0][0] == (0.65, 0.95, 'init_wl_delta: 0.4')
    assert mock_plt.text.call_args_list[1][0] == (0.65, 0.9, 'wl_scale: 0.8')
    assert mock_plt.text.call_args_list[2][0] == (0.65, 0.85, 'wl_delta_cutoff: 0.3')
