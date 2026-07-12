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
Unit tests for the module clustering.py.
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, call, MagicMock
from ensemble_md.analysis import clustering
from ensemble_md.analysis import analyze_traj


@patch('ensemble_md.analysis.clustering.analyze_transitions')
@patch('ensemble_md.analysis.clustering.get_cluster_members')
@patch('ensemble_md.analysis.clustering.get_cluster_info')
@patch('ensemble_md.analysis.clustering.run_gmx_cmd')
def test_cluster_traj(mock_gmx, mock_fn_1, mock_fn_2, mock_fn_3, capfd):
    # mock_fn_1 mocks get_cluster_info, mock_fn_2 mocks get_cluster_members, mock_fn_3 mocks analyze_transitions
    mock_gmx.return_value = 0, MagicMock(), MagicMock()  # returncode, stdout, stderr
    mock_fn_1.return_value = ([1.861, 6.340], 4.379, 2)  # rmsd_range, avg_rmsd, n_clusters
    mock_fn_2.return_value = ({1: [0, 2, 4], 2: [6, 8]}, {1: 0.6, 2: 0.4})  # clusters, sizes
    mock_fn_3.return_value = (np.array([[1, 2], [3, 1]]), np.array([1, 1, 1, 2, 2]), {(1, 2): 10})  # transitions, traj, transitions_dict (the values here doe not need to make sense)  # noqa: E501

    inputs = {
        'traj': 'ensemble_md/tests/data/traj.xtc',
        'config': 'ensemble_md/tests/data/sys.gro',
        'xvg': 'ensemble_md/tests/data/traj.xvg',
        'index': None
    }
    grps = {
        'center': 'HOS_MOL',
        'rmsd': 'complex_heavy',
        'output': 'HOS_MOL'
    }

    # Test 1: inputs not complete
    with pytest.raises(ValueError, match='The key "traj" is missing in the inputs dictionary.'):
        clustering.cluster_traj('gmx', {}, grps, coupled_only=True, cutoff=0.13, suffix='test')

    # Test 2: grps not complete
    with pytest.raises(ValueError, match='The key "center" is missing in the grps dictionary.'):
        clustering.cluster_traj('gmx', inputs, {}, coupled_only=True, cutoff=0.13, suffix='test')

    # Test 3: coupled_only is True but no xvg file is provided
    inputs['xvg'] = None
    with pytest.raises(ValueError, match='The parameter "coupled_only" is set to True but no XVG file is provided.'):
        clustering.cluster_traj('gmx', inputs, grps, coupled_only=True, cutoff=0.13, suffix='test')
    inputs['xvg'] = 'ensemble_md/tests/data/traj.xvg'

    # Test 4: No index file is provided
    mock_gmx.reset_mock()
    args = ['gmx', 'make_ndx', '-f', 'ensemble_md/tests/data/sys.gro', '-o', 'index.ndx']
    with pytest.raises(FileNotFoundError, match="No such file or directory: 'index.ndx'"):
        # We do not really run GROMACS commands so no index.ndx will be generated.
        # Still, we reach our goal to test the conditional block when inputs['index'] is None.
        clustering.cluster_traj('gmx', inputs, grps, coupled_only=True, cutoff=0.13, suffix='test')
    mock_gmx.assert_called_once_with(args, prompt_input='q\n')

    # Test 4b: gmx make_ndx itself fails (non-zero return code) -> should raise a clear
    # ValueError immediately instead of silently proceeding to a later, harder-to-diagnose crash.
    # (The other four run_gmx_cmd call sites in cluster_traj follow this exact same
    # "if returncode != 0: raise ValueError(...)" pattern, so this one case covers all of them.)
    mock_gmx.reset_mock()
    inputs['index'] = None  # Test 4's successful mocked make_ndx call already set this to 'index.ndx'  # noqa: E501
    mock_gmx.return_value = (1, MagicMock(), 'some gromacs error')
    with pytest.raises(ValueError, match='Error with return code 1'):
        clustering.cluster_traj('gmx', inputs, grps, coupled_only=True, cutoff=0.13, suffix='test')
    mock_gmx.return_value = (0, MagicMock(), MagicMock())  # restore for the remaining tests

    # Test 5: An index file is provided but groups of interest are not found
    mock_gmx.reset_mock()
    inputs['index'] = 'ensemble_md/tests/data/sys.ndx'
    grps['center'] = 'test'
    with pytest.raises(ValueError, match='The group "test" is not present in the provided/generated index file.'):
        clustering.cluster_traj('gmx', inputs, grps, coupled_only=True, cutoff=0.13, suffix='test')
    grps['center'] = 'HOS_MOL'

    # Test 6: An index file is provided but no coupled configurations are found
    mock_gmx.reset_mock()
    inputs['index'] = 'ensemble_md/tests/data/sys.ndx'
    inputs['xvg'] = 'traj_0.xvg'
    analyze_traj.convert_npy2xvg([np.ones(26)], 2)

    clustering.cluster_traj('gmx', inputs, grps, coupled_only=True, cutoff=0.13, suffix='test')
    out, err = capfd.readouterr()
    assert 'Terminating clustering analysis since no fully decoupled state is present in the input trajectory while coupled_only is set to True.' in out  # noqa: E501

    # Test 7: An index file is provided and there are coupled configurations
    mock_gmx.reset_mock()

    # save a rmsd_test.xvg just to avoid FileNotFoundError
    with open('rmsd_test.xvg', 'w') as f:
        f.write('0 1\n1 1\n2 1\n3 1\n4 1\n5 1\n')

    inputs['xvg'] = 'ensemble_md/tests/data/traj.xvg'
    clustering.cluster_traj('gmx', inputs, grps, coupled_only=True, cutoff=0.13, suffix='test')

    args_1 = [
        'gmx', 'trjconv',
        '-f', 'ensemble_md/tests/data/traj.xtc',
        '-s', 'ensemble_md/tests/data/sys.gro',
        '-n', 'ensemble_md/tests/data/sys.ndx',
        '-o', 'nojump_test.xtc',
        '-center', 'yes',
        '-pbc', 'nojump',
        '-drop', 'ensemble_md/tests/data/traj.xvg',
        '-dropover', '0'
    ]
    args_2 = [
        'gmx', 'trjconv',
        '-f', 'nojump_test.xtc',
        '-s', 'ensemble_md/tests/data/sys.gro',
        '-n', 'ensemble_md/tests/data/sys.ndx',
        '-o', 'center_test.xtc',
        '-center', 'yes',
        '-pbc', 'mol',
        '-ur', 'compact',
    ]
    args_3 = [
        'gmx', 'cluster',
        '-f', 'center_test.xtc',
        '-s', 'ensemble_md/tests/data/sys.gro',
        '-n', 'ensemble_md/tests/data/sys.ndx',
        '-o', 'rmsd_clust_test.xpm',
        '-dist', 'rmsd_dist_test.xvg',
        '-g', 'cluster_test.log',
        '-cl', 'clusters_test.pdb',
        '-cutoff', '0.13',
        '-method', 'linkage',
    ]
    args_4 = [
        'gmx', 'rms',
        '-f', 'clusters_test.pdb',
        '-s', 'clusters_test.pdb',
        '-o', 'rmsd_test.xvg',
        '-n', 'ensemble_md/tests/data/sys.ndx',
    ]

    assert mock_gmx.call_count == 4
    assert mock_fn_1.call_count == 1
    assert mock_fn_2.call_count == 1
    assert mock_fn_3.call_count == 1

    assert mock_gmx.call_args_list[0] == call(args_1, prompt_input='HOS_MOL\nHOS_MOL\n')
    assert mock_gmx.call_args_list[1] == call(args_2, prompt_input='HOS_MOL\nHOS_MOL\n')
    assert mock_gmx.call_args_list[2] == call(args_3, prompt_input='complex_heavy\nHOS_MOL\n')
    assert mock_gmx.call_args_list[3] == call(args_4, prompt_input='complex_heavy\ncomplex_heavy\n')

    out, err = capfd.readouterr()
    assert 'Number of fully coupled configurations: 5' in out
    assert 'Range of RMSD values: from 1.861 to 6.340 nm' in out
    assert 'Average RMSD: 4.379 nm' in out
    assert 'Number of clusters: 2' in out
    assert '  - Cluster 1 accounts for 60.00% of the total configurations.' in out
    assert '  - Cluster 2 accounts for 40.00% of the total configurations.' in out
    assert 'Inter-medoid RMSD between the two biggest clusters: 1.000 nm' in out

    os.remove('rmsd_test.xvg')
    os.remove('traj_0.xvg')


def test_get_cluster_info():
    cluster_log = 'ensemble_md/tests/data/cluster.log'
    results = clustering.get_cluster_info(cluster_log)

    assert results[0] == [0.0236461, 0.316756]
    assert results[1] == 0.182848
    assert results[2] == 2


def test_get_cluster_members():
    cluster_log = 'ensemble_md/tests/data/cluster.log'
    clusters, sizes = clustering.get_cluster_members(cluster_log)
    assert clusters == {
        1: [0, 178, 184, 186, 300, 302, 304, 306, 308, 310, 312, 318, 362, 366, 370, 372, 374, 376, 378, 380, 382, 390, 460, 464, 468, 470, 476],  # noqa: E501
        2: [11910, 11992, 11996, 12014, 12054, 12058, 12062, 12064, 12084, 12092, 12098, 12100, 12102, 12104, 12106, 12108, 12110, 12112, 12114, 12116, 12118, 12120, 12242, 12262, 12310, 12318, 12330, 12334, 12340],  # noqa: E501
    }
    assert sizes == {1: 27/56, 2: 29/56}


@patch('ensemble_md.analysis.clustering.plt')
def test_analyze_transitions(mock_plt):
    # Test 1: No transitions
    clusters = {
        1: [0, 178, 184, 186, 300, 302, 304, 306, 308, 310, 312, 318, 362, 366, 370, 372, 374, 376, 378, 380, 382, 390, 460, 464, 468, 470, 476],  # noqa: E501
    }
    results = clustering.analyze_transitions(clusters)
    assert results[0] == np.array([[1]])
    assert (results[1] == np.ones(len(clusters[1]))).all()
    assert results[2] == {}

    # Test 2: More transitions
    clusters = {
        1: [0, 2, 4, 6, 20, 22, 24, 34, 36, 38, 62, 64, 66, 80, 82, 84, 86],
        2: [8, 10, 12, 26, 28, 30, 32, 48, 50, 52, 54, 74, 76, 78, 88, 90, 92, 94, 96],
        3: [14, 16, 18, 40, 42, 44, 46, 56, 58, 60, 68, 70, 72],
    }
    results = clustering.analyze_transitions(clusters, normalize=False)
    traj = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 1, 1, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2]  # noqa: E501
    assert np.sum(results[0]) == sum([len(clusters[i]) for i in clusters]) - 1
    assert (results[0] == np.array([[12, 3, 2], [2, 14, 2], [2, 2, 9]])).all()
    assert (results[1] == traj).all()
    assert results[2] == {
        (1, 2): [8, 26, 34, 80, 88],
        (2, 3): [14, 48, 56, 74],
        (1, 3): [20, 40, 62, 68],
    }

    # Test 3: Test the plotting functionalities
    # 3-1. plt_type = 'bar'
    mock_ax, mock_fig, mock_obj = MagicMock(), MagicMock(), MagicMock()
    mock_plt.figure.return_value = mock_fig
    mock_fig.add_subplot.return_value = mock_ax
    mock_plt.MaxNLocator.return_value = mock_obj

    results = clustering.analyze_transitions(clusters, plot_type='bar')

    mock_plt.figure.called_once()
    mock_fig.add_subplot.assert_called_once_with(111)
    mock_plt.bar.called_once()
    mock_plt.xlabel.assert_called_once_with('Cluster index')
    mock_plt.ylabel.assert_called_once_with('Number of configurations')
    mock_plt.grid.assert_called_once()
    mock_ax.xaxis.set_major_locator.assert_called_once_with(mock_obj)
    mock_ax.yaxis.set_major_locator.assert_called_once_with(mock_obj)
    mock_plt.savefig.assert_called_once_with('cluster_distribution.png', dpi=600)

    assert list(mock_plt.bar.call_args_list[0][0][0]) == [1, 2, 3]
    assert mock_plt.bar.call_args_list[0][0][1] == [17, 19, 13]
    assert mock_plt.bar.call_args_list[0][1] == {'width': 0.35}

    # 3-2. plt_type = 'xy', short traj
    mock_plt.reset_mock()

    results = clustering.analyze_transitions(clusters, plot_type='xy')

    mock_plt.figure.called_once()
    mock_fig.add_subplot.assert_called_once_with(111)
    mock_plt.plot.called_once()
    mock_plt.xlabel.assert_called_once_with('Time frame (ps)')
    mock_plt.ylabel.assert_called_once_with('Cluster index')
    mock_ax.yaxis.set_major_locator.assert_called_once_with(mock_obj)
    mock_plt.grid.assert_called_once()
    mock_plt.savefig.assert_called_once_with('cluster_traj.png', dpi=600)

    assert (mock_plt.plot.call_args_list[0][0][0] == np.arange(0, 98, 2)).all()
    assert (mock_plt.plot.call_args_list[0][0][1] == traj).all()

    # 3-3. plt_type = 'xy', long traj
    mock_plt.reset_mock()
    clusters = {}
    t = np.arange(2000)
    clusters[1] = np.random.choice(t, 1000, replace=False)
    clusters[1] = np.sort(clusters[1])
    clusters[2] = np.array([t[i] for i in range(2000) if i not in clusters[1]])

    results = clustering.analyze_transitions(clusters, plot_type='xy')
    mock_plt.xlabel.assert_called_once_with('Time frame (ns)')
    assert (mock_plt.plot.call_args_list[0][0][0] == np.arange(2000) / 1000).all()

    # 3-4. invalid plt_type
    with pytest.raises(ValueError, match='Invalid plot type: test. The plot type must be either "bar" or "xy" or unspecified.'):  # noqa: E501
        clustering.analyze_transitions(clusters, plot_type='test')
