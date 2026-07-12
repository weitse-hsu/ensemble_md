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
Unit tests for the module utils.py.
"""
import os
import sys
import shutil
import pytest
import tempfile
import subprocess
import numpy as np
from ensemble_md.utils import utils
from unittest.mock import patch, MagicMock


def test_logger():
    # Create a temporary file for the log
    with tempfile.TemporaryFile(mode="w+t") as log_file:
        # Get the file path for the temporary file
        log_path = log_file.name

        # Create a logger that redirects output to the temporary file
        logger = utils.Logger(log_path)

        # Redirect stdout to the logger
        sys.stdout = logger

        # Write some messages to stdout
        print("Hello, world!")
        print("Testing logger...")

        # Flush the logger to ensure that all messages are written to the log
        logger.flush()

        # Reset stdout to the original stream
        sys.stdout = sys.__stdout__

        # Logger.flush() is a no-op, so flush the underlying file object directly to make sure
        # the written content is actually visible before reading it back.
        logger.log.flush()
        log_file.seek(0)
        content = log_file.read()
        assert "Hello, world!" in content
        assert "Testing logger..." in content


def test_run_gmx_cmd_success():
    # Mock the subprocess.run return value for a successful execution
    mock_successful_return = MagicMock()
    mock_successful_return.returncode = 0
    mock_successful_return.stdout = "Simulation complete"
    mock_successful_return.stderr = None

    with patch('subprocess.run', return_value=mock_successful_return) as mock_run:
        return_code, stdout, stderr = utils.run_gmx_cmd(['gmx', 'mdrun', '-deffnm', 'sys'])

    mock_run.assert_called_once_with(['gmx', 'mdrun', '-deffnm', 'sys'], capture_output=True, text=True, input=None, check=True)  # noqa: E501
    assert return_code == 0
    assert stdout == "Simulation complete"
    assert stderr is None


def test_run_gmx_cmd_failure():
    # Mock the subprocess.run to raise a CalledProcessError for a failed execution
    mock_failed_return = MagicMock()
    mock_failed_return.returncode = 1
    mock_failed_return.stderr = "Error encountered"

    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = [subprocess.CalledProcessError(mock_failed_return.returncode, 'cmd', stderr=mock_failed_return.stderr)]  # noqa: E501
        return_code, stdout, stderr = utils.run_gmx_cmd(['gmx', 'mdrun', '-deffnm', 'sys'])

    assert return_code == 1
    assert stdout is None
    assert stderr == "Error encountered"


def test_format_time():
    assert utils.format_time(0) == "0.0 second(s)"
    assert utils.format_time(1) == "1.0 second(s)"
    assert utils.format_time(59) == "59.0 second(s)"
    assert utils.format_time(60) == "1 minute(s) 0 second(s)"
    assert utils.format_time(61) == "1 minute(s) 1 second(s)"
    assert utils.format_time(3599) == "59 minute(s) 59 second(s)"
    assert utils.format_time(3600) == "1 hour(s) 0 minute(s) 0 second(s)"
    assert utils.format_time(3661) == "1 hour(s) 1 minute(s) 1 second(s)"
    assert utils.format_time(86399) == "23 hour(s) 59 minute(s) 59 second(s)"
    assert utils.format_time(86400) == "1 day, 0 hour(s) 0 minute(s) 0 second(s)"
    assert utils.format_time(90061) == "1 day, 1 hour(s) 1 minute(s) 1 second(s)"


def test_convert_to_numeric():
    # Test non-string input
    assert utils._convert_to_numeric(42) == 42
    assert utils._convert_to_numeric("42") == 42
    assert utils._convert_to_numeric("3.14159") == 3.14159
    assert utils._convert_to_numeric("1 2 3") == [1, 2, 3]
    assert utils._convert_to_numeric("1.0 2.0 3.0") == [1.0, 2.0, 3.0]
    assert utils._convert_to_numeric("Hello, world!") == ['Hello,', 'world!']
    assert utils._convert_to_numeric('Y Y Y') == ['Y', 'Y', 'Y']


def test_get_subplot_dimension():
    assert utils._get_subplot_dimension(1) == (1, 1)
    assert utils._get_subplot_dimension(2) == (1, 2)
    assert utils._get_subplot_dimension(3) == (2, 2)
    assert utils._get_subplot_dimension(4) == (2, 2)
    assert utils._get_subplot_dimension(5) == (2, 3)
    assert utils._get_subplot_dimension(6) == (2, 3)
    assert utils._get_subplot_dimension(7) == (3, 3)
    assert utils._get_subplot_dimension(8) == (3, 3)
    assert utils._get_subplot_dimension(9) == (3, 3)


def test_weighted_mean():
    # 1. Same error for each sample --> weighted mean reduces to simple mean
    vals = [1, 2, 3, 4]
    errs = [0.1, 0.1, 0.1, 0.1]
    mean, err = utils.weighted_mean(vals, errs)
    assert mean == 2.5
    assert err == np.sqrt(4 * 0.1 ** 2) / 4  # 0.05

    # 2. Different errors
    vals = [1, 2, 3, 4]
    errs = [5, 0.1, 0.1, 0.1]
    mean, err = utils.weighted_mean(vals, errs)
    assert np.isclose(mean, 2.9997333688841485)
    assert np.isclose(err, 0.0577311783020254)

    # 3. 0 in errs
    vals = [1, 2, 3, 4]
    errs = [0, 0.1, 0.1, 0.1]
    mean, err = utils.weighted_mean(vals, errs)
    assert mean == 2.5
    assert err is None


def test_calc_rmse():
    # Test 1
    data = [1, 2, 3, 4, 5]
    ref = [2, 4, 6, 8, 10]
    expected_rmse = np.sqrt(np.mean((np.array(data) - np.array(ref)) ** 2))
    assert utils.calc_rmse(data, ref) == expected_rmse

    # Test 2
    ref = [1, 2, 3, 4, 5]
    expected_rmse = 0
    assert utils.calc_rmse(data, ref) == expected_rmse

    # Test 3
    data = [1, 2, 3]
    ref = [1, 2]
    with pytest.raises(ValueError):
        utils.calc_rmse(data, ref)


def test_get_time_metrics():
    log = 'ensemble_md/tests/data/log/EXE.log'
    t_metrics = {
        'performance': 23.267,
        't_wall': 3.721,
        't_core': 29.713
    }
    assert utils.get_time_metrics(log) == t_metrics


def test_analyze_REXEE_time():
    # Set up directories and files
    dirs = [f'ensemble_md/tests/data/log/sim_{i}/iteration_{j}' for i in range(2) for j in range(2)]
    files = [f'ensemble_md/tests/data/log/EXE_{i}.log' for i in range(4)]
    for i in range(4):
        os.makedirs(dirs[i])
        shutil.copy(files[i], os.path.join(dirs[i], 'EXE.log'))

    # Test analyze_REXEE_time
    # Case 1: Wrong paths
    with pytest.raises(FileNotFoundError, match="No sim/iteration directories found."):
        t_1, t_2, t_3 = utils.analyze_REXEE_time()  # This will try to find files from [natsort.natsorted(glob.glob(f'sim_*/iteration_{i}/*log')) for i in range(n_iter)]  # noqa: E501

    # Case 2: Correct paths
    log_files = [[f'ensemble_md/tests/data/log/sim_{i}/iteration_{j}/EXE.log' for i in range(2)] for j in range(2)]
    t_1, t_2, t_3 = utils.analyze_REXEE_time(log_files=log_files)
    assert t_1 == 2.125
    assert np.isclose(t_2, 0.175)
    assert t_3 == [[1.067, 0.94], [1.01, 1.058]]

    # Clean up
    for i in range(2):
        shutil.rmtree(f'ensemble_md/tests/data/log/sim_{i}')
