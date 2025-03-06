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
The :obj:`.replica_exchange_EE` module provides functions for setting up and
performing replica exchange and expanded ensemble (REXEE) simulations.
"""
import os
import sys
import copy
import yaml
import random
import warnings
import importlib
import subprocess
import mdtraj as md
import pandas as pd
import numpy as np
from mpi4py import MPI
from itertools import combinations
from collections import OrderedDict
from alchemlyb.parsing.gmx import _get_headers as get_headers
from alchemlyb.parsing.gmx import _extract_dataframe as extract_dataframe

import ensemble_md
from ensemble_md.utils import utils
from ensemble_md.utils import gmx_parser
from ensemble_md.utils import coordinate_swap
from ensemble_md.utils.exceptions import ParameterError

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class ReplicaExchangeEE:
    """
    A class that provides a variety of functions useful for setting up and running
    a replica exchange (REX) of expanded ensemble (EE) simulation, or a REXEE simulation.
    Upon instantiation, all parameters in the input YAML file will be assigned to an
    attribute in the class. In addition to these variables, below is a list of
    attributes of the class. (All the the attributes are assigned by :obj:`set_params`
    except that :code:`yaml` is assigned by :code:`__init__`.)

    :ivar gmx_path: The absolute path of the GROMACS exectuable.
    :ivar gmx_version: The version of the GROMACS executable.
    :ivar yaml: The input YAML file used to instantiate the class. The file should contain necessary REXEE parameters.
        For more details, please check the :ref:`doc_parameters`.
    :ivar warnings: Warnings about parameter specification in either YAML or MDP files.
    :ivar template: The template MDP file on which the instance of the :obj:`MDP` class is based.
    :ivar reformatted_mdp: Whether the template MDP file has been reformatted by replacing hyphens
        with underscores or not.
    :ivar dt: The simulation timestep in ps.
    :ivar temp: The simulation temperature in Kelvin.
    :ivar fixed_weights: Whether the weights will be fixed during the simulation.
    :ivar updating_weights: The list of weights as a function of time (since the last update of the Wang-Landau
        incrementor) for different replicas. The length is equal to the number of replicas. This is only relevant for
        weight-updating simulations.
    :ivar equilibrated_weights: The equilibrated weights of different replicas. For weight-updating simulations,
        this list is initialized as a list of empty lists. Otherwise (i.e., in fixed-weight simulations), it is
        initialized as a list of :code:`None`.
    :ivar current_wl_delta: The current value of the Wang-Landau incrementor. This is only relevent for weight-updating
        simulations.
    :ivar kT: 1 kT in kJ/mol at the simulation temperature.
    :ivar lambda_types: The types of lambda variables involved in expanded ensemble simulations, e.g.,
        :code:`fep_lambdas`, :code:`mass_lambdas`, :code:`coul_lambdas`, etc.
    :ivar n_tot: The total number of states for all replicas.
    :ivar n_sub: The numbmer of states of each replica. The current implementation assumes homogenous replicas.
    :ivar state_ranges: A list of list of (global) state indices for each replica.
    :ivar equil: A list of times it took to equilibrate the weights for different replicas. This
        list is initialized with a list of -1, where -1 means that the weights haven't been equilibrated. Also,
        a value of 0 means that the simulation is a fixed-weight simulation.
    :ivar n_rejected: The number of proposed exchanges that have been rejected. Updated by :obj:`.accept_or_reject`.
    :ivar n_swap_attempts: The number of swaps attempted so far. This does not include the cases
        where there is no swappable pair. Updated by :obj:`.get_swapping_pattern`.
    :ivar n_emtpy_swappable: The number of times when there was no swappable pair.
    :ivar rep_trajs: The replica-space trajectories of all configurations.
    :ivar configs: A list that thows the current configuration index that each replica is sampling.
    :ivar g_vecs: The time series of processed (e.g., combined across replicas) alchemical weights for the entire state
        space. If no weight combination scheme is applied, this list will just be a list of :code:`None`'s.
    :ivar df_data_type: The type of data (either :math:`u_{nk}` or :math:`dH/dλ`) that will be used for
        free energy calculations. This depends on the free energy estimator specified in the parameter
        :code:`df_method`.
    :ivar modify_coords_fn: The function (callable) in an external module (specified as :code:`modify_coords` in
        the input YAML file) for modifying coordinates at exchanges. This parameter is only relevant to
        multi-topology REXEE (i.e., MT-REXEE) simulations.
    """

    def __init__(self, yaml_file, analysis=False):
        self.yaml = yaml_file
        self.set_params(analysis)

    def set_params(self, analysis):
        """
        Sets up or reads in the user-defined parameters from an input YAML file and an MDP template.
        This function is called to instantiate the class in the :code:`__init__` function of
        class. Specifically, it does the following:

          1. Sets up constants.
          2. Reads in REXEE parameters from a YAML file.
          3. Handles YAML parameters.
          4. Checks if the parameters in the YAML file are well-defined.
          5. Reformats the input MDP file to replace all hyphens with underscores.
          6. Reads in parameters from the MDP template.

        After instantiation, the class instance will have an attribute corresponding to
        each of the parameters specified in the YAML file. For a full list of the parameters that
        can be specified in the YAML file, please refer to :ref:`doc_parameters`.

        :param yaml_file: The file path of the input YAML file that specifies REXEE parameters.
        :type yaml_file: str
        :param analysis: Whether the instantiation of the class is for data analysis of REXEE simulations.
            The default is :code:`False`.
        :type analysis: bool, Optional

        :raises ParameterError:

              - If a required parameter is not specified in the input YAML file.
              - If a specified parameter is not recognizable.
              - If a specified option is not available for a parameter.
              - If the data type or range (e.g., positive or negative) of a parameter is not correct.
              - If any MDP parameter invalid for the REXEE simulation is detected.
        """
        self.warnings = []  # Store warnings, if any.

        # Step 0: Set up constants
        k = 1.380649e-23
        NA = 6.0221408e23

        # Step 1: Read in parameters from the YAML file.
        with open(self.yaml) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        for attr in params:
            setattr(self, attr, params[attr])

        # Step 2: Handle the compulsory YAML parameters
        required_args = [
            "gmx_executable",
            "gro",
            "top",
            "mdp",
            "n_sim",
            "n_iter",
            "s",
        ]
        for i in required_args:
            if hasattr(self, i) is False or getattr(self, i) is None:
                raise ParameterError(
                    f"Required parameter '{i}' not specified in {self.yaml}."
                )  # noqa: F405

        # Step 3: Handle the optional YAML parameters
        # Key: Optional argument; Value: Default value
        optional_args = {
            "add_swappables": None,
            "modify_coords": None,
            "resname_list": None,
            "swap_rep_pattern": None,
            "nst_sim": None,
            "proposal": 'exhaustive',
            "w_combine": False,
            "w_mean_type": 'simple',
            "N_cutoff": -1,
            "hist_corr": False,
            "verbose": True,
            "mdp_args": None,
            "grompp_args": None,
            "runtime_args": None,
            "n_ckpt": 100,
            "rm_cpt": True,
            "msm": False,
            "free_energy": False,
            "subsampling_avg": False,
            "df_spacing": 1,
            "df_ref": None,
            "df_method": "MBAR",
            "err_method": "propagate",
            "n_bootstrap": 50,
            "seed": None,
            # "n_ex": 'N^3',   # only active for multiple swaps.
        }
        for i in optional_args:
            if hasattr(self, i) is False or getattr(self, i) is None:
                setattr(self, i, optional_args[i])

        # all_args: Arguments that can be specified in the YAML file.
        all_args = required_args + list(optional_args.keys())
        for i in params:
            if i not in all_args:
                self.warnings.append(f'Warning: Parameter "{i}" specified in the input YAML file is not recognizable.')

        # Step 4: Check if the parameters in the YAML file are well-defined
        if self.proposal not in [None, 'single', 'neighboring', 'exhaustive', 'random_range']:  # deprecated option: multiple  # noqa: E501
            raise ParameterError("The specified proposal scheme is not available. Available options include 'single', 'neighboring', 'exhaustive', 'random_range'.")  # noqa: E501

        if self.df_method not in [None, 'TI', 'BAR', 'MBAR']:
            raise ParameterError("The specified free energy estimator is not available. Available options include 'TI', 'BAR', and 'MBAR'.")  # noqa: E501

        if self.err_method not in [None, 'propagate', 'bootstrap']:
            raise ParameterError("The specified method for error estimation is not available. Available options include 'propagate', and 'bootstrap'.")  # noqa: E501

        params_int = ['n_sim', 'n_iter', 's', 'N_cutoff', 'df_spacing', 'n_ckpt', 'n_bootstrap']  # integer parameters  # noqa: E501
        if self.nst_sim is not None:
            params_int.append('nst_sim')
        """
        if self.n_ex != 'N^3':  # no need to add "and self.proposal == 'multiple' since if multiple swaps are not used, n_ex=1"  # noqa: E501
            params_int.append('n_ex')
        """
        if self.seed is not None:
            params_int.append('seed')
        for i in params_int:
            if type(getattr(self, i)) != int:
                raise ParameterError(f"The parameter '{i}' should be an integer.")

        params_pos = ['n_sim', 'n_iter', 'n_ckpt', 'df_spacing', 'n_bootstrap']  # positive parameters
        if self.nst_sim is not None:
            params_pos.append('nst_sim')
        for i in params_pos:
            if getattr(self, i) <= 0:
                raise ParameterError(f"The parameter '{i}' should be positive.")

        """
        if self.n_ex != 'N^3' and self.n_ex < 0:
            raise ParameterError("The parameter 'n_ex' should be non-negative.")
        """

        if self.s < 0:
            raise ParameterError("The parameter 's' should be non-negative.")

        if self.N_cutoff < 0 and self.N_cutoff != -1:
            raise ParameterError("The parameter 'N_cutoff' should be non-negative unless no weight correction is needed, i.e. N_cutoff = -1.")  # noqa: E501

        params_str = ['gro', 'top', 'mdp', 'gmx_executable']
        # First check if self.gro and self.top are lists and check their lengths
        check_files = ['gro', 'top']  # just for checking file types that can take multiple inputs
        for i in check_files:
            if isinstance(getattr(self, i), list):
                params_str.remove(i)
                if len(getattr(self, i)) != self.n_sim:
                    raise ParameterError(f"The number of the input {i.upper()} files must be the same as the number of replicas, if multiple are specified.")  # noqa: E501
        if self.modify_coords is not None:
            params_str.append('modify_coords')
        for i in params_str:
            if type(getattr(self, i)) != str:
                raise ParameterError(f"The parameter '{i}' should be a string.")
        self.working_dir = os.path.dirname(self.mdp)
        if self.working_dir == '':
            self.working_dir = '.'

        params_bool = ['verbose', 'rm_cpt', 'msm', 'free_energy', 'subsampling_avg', 'w_combine']
        for i in params_bool:
            if type(getattr(self, i)) != bool:
                raise ParameterError(f"The parameter '{i}' should be a boolean variable.")

        params_list = ['add_swappables', 'df_ref']
        if self.resname_list is not None:
            params_list.append('resname_list')
        if self.swap_rep_pattern is not None:
            params_list.append('swap_rep_pattern')
        for i in params_list:
            if getattr(self, i) is not None and not isinstance(getattr(self, i), list):
                raise ParameterError(f"The parameter '{i}' should be a list.")

        params_dict = ['mdp_args', 'grompp_args', 'runtime_args']
        for i in params_dict:
            if getattr(self, i) is not None and not isinstance(getattr(self, i), dict):
                raise ParameterError(f"The parameter '{i}' should be a dictionary.")

        if self.add_swappables is not None:
            for sublist in self.add_swappables:
                if not isinstance(sublist, list):
                    raise ParameterError("The parameter 'add_swappables' should be a nested list.")
                for item in sublist:
                    if not isinstance(item, int) or item < 0:
                        raise ParameterError("Each number specified in 'add_swappables' should be a non-negative integer.")  # noqa: E501
            if [len(i) for i in self.add_swappables] != [2] * len(self.add_swappables):
                raise ParameterError("Each sublist in 'add_swappables' should contain two integers.")

        if self.mdp_args is not None:
            # Note that mdp_args is a dictionary including MDP parameters DIFFERING across replicas.
            # The value of each key should be a list of length n_sim.
            for val in self.mdp_args.values():
                if not isinstance(val, list):
                    raise ParameterError("The values specified in 'mdp_args' should be lists.")

                if len(set(val)) == 1:
                    raise ParameterError("MDP parameters set by 'mdp_args' should differ across at least two replicas.")  # noqa: E501

            for key in self.mdp_args.keys():
                if not isinstance(key, str):
                    raise ParameterError("All keys specified in 'mdp_args' should be strings.")
                else:
                    if '-' in key:
                        raise ParameterError("ensemble_md convention: Parameters specified in 'mdp_args' must use underscores in place of hyphens.")  # noqa: E501
            for val_list in self.mdp_args.values():
                if len(val_list) != self.n_sim:
                    raise ParameterError("The number of values specified for each key in 'mdp_args' should be the same as the number of replicas.")  # noqa: E501

        # Step 5: Reformat the input MDP file to replace all hypens with underscores.
        self.reformatted_mdp = ReplicaExchangeEE.reformat_MDP(self.mdp)

        # Step 6: Read in parameters from the MDP template
        self.template = gmx_parser.MDP(self.mdp)
        self.dt = self.template["dt"]  # ps
        self.temp = self.template["ref_t"]

        if self.nst_sim is None:
            self.nst_sim = self.template["nsteps"]

        if 'wl_scale' in self.template.keys():
            if self.template['wl_scale'] == []:
                # If wl_scale in the MDP file is a blank (i.e. fixed weights), mdp['wl_scale'] will be an empty list.
                # This is the only case where mdp['wl_scale'] is a numpy array.
                self.fixed_weights = True
                self.equilibrated_weights = [None for i in range(self.n_sim)]
            else:
                self.fixed_weights = False
                self.equilibrated_weights = [[] for i in range(self.n_sim)]
                self.updating_weights = [[] for i in range(self.n_sim)]
                self.current_wl_delta = [0 for i in range(self.n_sim)]
        else:
            self.fixed_weights = True
            self.equilibrated_weights = [None for i in range(self.n_sim)]

        if self.fixed_weights is True:
            if self.N_cutoff != -1 or self.w_combine is True:
                self.warnings.append('Warning: The weight correction/weight combination method is specified but will not be used since the weights are fixed.')  # noqa: E501
                # In the case that the warning is ignored, enforce the defaults.
                self.N_cutoff = -1
                self.w_combine = False

        if 'lmc_seed' in self.template and self.template['lmc_seed'] != -1:
            self.warnings.append('Warning: We recommend setting lmc_seed as -1 so the random seed is different for each iteration.')  # noqa: E501

        if 'gen_seed' in self.template and self.template['gen_seed'] != -1:
            self.warnings.append('Warning: We recommend setting gen_seed as -1 so the random seed is different for each iteration.')  # noqa: E501

        if 'gen_vel' not in self.template or ('gen_vel' in self.template and self.template['gen_vel'] == 'no'):
            self.warnings.append('Warning: We recommend generating new velocities for each iteration to avoid potential issues with the detailed balance.')  # noqa: E501

        if self.nst_sim % self.template['nstlog'] != 0:
            raise ParameterError(
                'The parameter "nstlog" must be a factor of the parameter "nst_sim" specified in the YAML file.')

        if self.nst_sim % self.template['nstdhdl'] != 0:
            raise ParameterError(
                'The parameter "nstdhdl" must be a factor of the parameter "nst_sim" specified in the YAML file.')

        if self.template['nstexpanded'] % self.template['nstdhdl'] != 0:
            raise ParameterError(
                'In REXEE, the parameter "nstdhdl" must be a factor of the parameter "nstexpanded", or the calculation of acceptance ratios may be wrong.')  # noqa: E501

        if self.mdp_args is not None:
            # Varying the following parameters may not make sense, but here we just avoid edge cases.
            # We check these parameters as they could directly influence the correctness of the simulation.
            if 'lmc_seed' in self.mdp_args and self.mdp_args['lmc_seed'] != [-1] * self.n_sim:
                self.warnings.append('Warning: We recommend setting lmc_seed as -1 so the random seed is different for each iteration.')  # noqa: E501

            if 'gen_seed' in self.mdp_args and self.mdp_args['gen_seed'] != [-1] * self.n_sim:
                self.warnings.append('Warning: We recommend setting gen_seed as -1 so the random seed is different for each iteration.')  # noqa: E501

            if 'gen_vel' in self.mdp_args and self.mdp_args['gen_vel'] != ['yes'] * self.n_sim:
                self.warnings.append('Warning: We recommend generating new velocities for each iteration to avoid potential issues with the detailed balance.')  # noqa: E501

            if 'nstlog' in self.mdp_args and sum(self.nst_sim % np.array(self.mdp_args['nstlog'])) != 0:
                raise ParameterError(
                    'The parameter "nstlog" must be a factor of the parameter "nst_sim" specified in the YAML file.')

            if 'nstdhdl' in self.mdp_args and sum(self.nst_sim % np.array(self.mdp_args['nstdhdl'])) != 0:
                raise ParameterError(
                    'The parameter "nstdhdl" must be a factor of the parameter "nst_sim" specified in the YAML file.')

            if 'nstexpanded' in self.mdp_args and 'nstdhdl' in self.mdp_args and sum(np.array(self.mdp_args['nstexpanded']) % np.array(self.mdp_args['nstdhdl'])) != 0:  # noqa: E501
                raise ParameterError(
                    'In REXEE, the parameter "nstdhdl" must be a factor of the parameter "nstexpanded", or the calculation of acceptance ratios may be wrong.')  # noqa: E501

        if 'pull' in self.template and self.template['pull'] == 'yes':
            pull_ncoords = self.template['pull_ncoords']
            self.set_ref_dist = []
            for i in range(pull_ncoords):
                if self.template[f'pull_coord{i+1}_geometry'] == 'distance':
                    if self.template[f'pull_coord{i+1}_start'] == 'yes':
                        self.set_ref_dist.append(True)  # starting from the second iteration, set pull_coord*_init.
                        if 'pull_nstxout' not in self.template or self.template['pull_nstxout'] == 0:
                            self.warnings.append('A non-zero value should be specified for pull_nstxout if pull_coord*_start is set to yes.')  # noqa: E501
                    else:
                        self.set_ref_dist.append(False)  # Here we assume that the user know what reference distance to use.  # noqa: E501
                else:
                    self.set_ref_dist.append(False)  # we only deal with distance restraints for now.

        # Step 7: Set up derived parameters
        # 7-1. kT in kJ/mol
        self.kT = k * NA * self.temp / 1000  # 1 kT in kJ/mol

        # 7-2. Figure out what types of lambda variables are involved
        # Here is we possible lambda types in the order read by GROMACS, which is likely also the order when being printed to the log file.  # noqa: E501
        # See https://gitlab.com/gromacs/gromacs/-/blob/main/src/gromacs/gmxpreprocess/readir.cpp#L2543
        lambdas_types_all = ['fep_lambdas', 'mass_lambdas', 'coul_lambdas', 'vdw_lambdas', 'bonded_lambdas', 'restraint_lambdas', 'temperature_lambdas']  # noqa: E501
        self.lambda_types = []  # lambdas specified in the MDP file
        for i in lambdas_types_all:
            if i in self.template.keys():  # there shouldn't be parameters like "fep-lambdas" after reformatting the MDP file  # noqa: E501
                self.lambda_types.append(i)

        # 7-3. Total # of states: n_tot = n_sub * n_sim - (n_overlap) * (n_sim - 1), where n_overlap = n_sub - s
        self.n_tot = len(self.template[self.lambda_types[0]])

        # 7-4. Number of states of each replica (assuming the same for all rep)
        self.n_sub = self.n_tot - self.s * (self.n_sim - 1)
        if self.n_sub < 1:
            raise ParameterError(
                f"There must be at least two states for each replica (current value: {self.n_sub}). The current specified configuration (n_tot={self.n_tot}, n_sim={self.n_sim}, s={self.s}) does not work for REXEE.")  # noqa: E501

        # 7-5. A list of sets of state indices
        start_idx = [i * self.s for i in range(self.n_sim)]
        self.state_ranges = [list(np.arange(i, i + self.n_sub, dtype=np.int32)) for i in start_idx]

        # 7-6. A list of time it took to get the weights equilibrated
        self.equil = [-1 for i in range(self.n_sim)]   # -1 means unequilibrated

        # 7-7. Some variables for counting
        self.n_rejected = 0
        self.n_swap_attempts = 0
        self.n_empty_swappable = 0

        # 7-8. Replica space trajectories. For example, rep_trajs[0] = [0, 2, 3, 0, 1, ...] means
        # that configuration 0 transitioned to replica 2, then 3, 0, 1, in iterations 1, 2, 3, 4, ...,
        # respectively. The first element of rep_traj[i] should always be i.
        self.rep_trajs = [[i] for i in range(self.n_sim)]

        # 7-9. configs shows the current configuration that each replica is sampling.
        # For example, self.configs = [0, 2, 1, 3] means that configurations 0, 2, 1, and 3 are
        # in replicas, 0, 1, 2, 3, respectively. This list will be constantly updated during the simulation.
        self.configs = list(range(self.n_sim))

        # 7-10. The time series of the (processed) whole-range alchemical weights
        # If no weight combination is applied, self.g_vecs will just be a list of None's.
        self.g_vecs = []

        # 7-11. Data analysis
        if self.df_method == 'MBAR':
            self.df_data_type = 'u_nk'
        else:
            self.df_data_type = 'dhdl'

        # 7-12. External module for coordinate modification
        if self.modify_coords is not None:
            if self.modify_coords == 'default':
                if self.swap_rep_pattern is None and (not os.path.exists('residue_connect.csv') or not os.path.exists('residue_swap_map.csv')):  # noqa: E501
                    raise ParameterError('swap_rep_pattern option must be filled in if using default swapping function and not swap guide')  # noqa: E501
                if self.resname_list is None and (not os.path.exists('residue_connect.csv') or not os.path.exists('residue_swap_map.csv')):  # noqa: E501
                    raise ParameterError('resname_list option must be filled in if using default swapping function and not swap guide')  # noqa: E501
                self.modify_coords_fn = self.default_coords_fn
            else:
                module_file = os.path.basename(self.modify_coords)
                module_dir = os.path.dirname(self.modify_coords)
                if module_dir not in sys.path:
                    sys.path.append(module_dir)  # so that the module can be imported
                module_name = os.path.splitext(module_file)[0]
                module = importlib.import_module(module_name)
                if not hasattr(module, module_name):
                    err_msg = f'The module for coordinate manipulation (specified through the parameter) must have a function with the same name as the module, i.e., {module_name}.'  # noqa: E501
                    raise ParameterError(err_msg)
                else:
                    self.modify_coords_fn = getattr(module, module_name)
        else:
            self.modify_coords_fn = None

        # Step 8. Check the executables
        if analysis is False:
            self.check_gmx_executable()

    def check_gmx_executable(self):
        """
        Checks if the GROMACS executable can be used and gets its absolute path and version.
        """
        try:
            result = subprocess.run(['which', self.gmx_executable], capture_output=True, text=True, check=True)
            self.gmx_path = result.stdout.strip()  # this can be exactly the same as self.gmx_executable

            version_output = subprocess.run([self.gmx_path, "-version"], capture_output=True, text=True, check=True)
            for line in version_output.stdout.splitlines():
                if "GROMACS version" in line:
                    self.gmx_version = line.split()[-1]
                    break
        except subprocess.CalledProcessError:
            print(f"{self.gmx_executable} is not available.")
        except Exception as e:
            print(f"An error occurred:\n{e}")

    def print_params(self, params_analysis=False):
        """
        Prints important parameters relevant to the REXEE simulation to be performed.

        Parameters
        ----------
        params_analysis : bool, Optional
            Whether additional parameters for data analysis should be printed. The default is :code:`False`.
        """
        if isinstance(self.gro, list):
            gro_str = ', '.join(self.gro)
        else:
            gro_str = self.gro

        if isinstance(self.top, list):
            top_str = ', '.join(self.top)
        else:
            top_str = self.top

        print("Important parameters of REXEE")
        print("=============================")
        print(f"Python version: {sys.version}")
        print(f"GROMACS executable: {self.gmx_path}")  # we print the full path here
        print(f"GROMACS version: {self.gmx_version}")
        print(f"ensemble_md version: {ensemble_md.__version__}")
        print(f'Simulation inputs: {gro_str}, {top_str}, {self.mdp}')
        print(f"Verbose log file: {self.verbose}")
        print(f"Proposal scheme: {self.proposal}")
        print(f"Whether to perform weight combination: {self.w_combine}")
        print(f"Type of means for weight combination: {self.w_mean_type}")
        print(f"Whether to perform histogram correction: {self.hist_corr}")
        print(f"Histogram cutoff for weight correction: {self.N_cutoff}")
        print(f"Number of replicas: {self.n_sim}")
        print(f"Number of iterations: {self.n_iter}")
        print(f"Length of each replica: {self.dt * self.nst_sim} ps")
        print(f"Frequency for checkpointing: {self.n_ckpt} iterations")
        print(f"Total number of states: {self.n_tot}")
        print(f"Additionally defined swappable states: {self.add_swappables}")
        print(f"Additional grompp arguments: {self.grompp_args}")
        print(f"Additional runtime arguments: {self.runtime_args}")
        print(f"External modules for coordinate manipulation: {self.modify_coords}")
        # print(f"Number of attempted swaps in one exchange interval: {self.n_ex}")
        if self.mdp_args is not None and len(self.mdp_args.keys()) > 1:
            print("MDP parameters differing across replicas:")
            for i in self.mdp_args.keys():
                print(f"  - {i}: {self.mdp_args[i]}")
        else:
            print(f"MDP parameters differing across replicas: {self.mdp_args}")
        print("Alchemical ranges of each replica in REXEE:")
        for i in range(self.n_sim):
            print(f"  - Replica {i}: States {self.state_ranges[i]}")

        if params_analysis is True:
            print()
            print(f"Whether to build Markov state models and perform relevant analysis: {self.msm}")
            print(f"Whether to perform free energy calculations: {self.free_energy}")
            print(f"The step to used in subsampling the DHDL data in free energy calculations, if any: {self.df_spacing}")  # noqa: E501
            print(f"The chosen free energy estimator for free energy calculations, if any: {self.df_method}")
            print(f"The method for estimating the uncertainty of free energies in free energy calculations, if any: {self.err_method}")  # noqa: E501
            print(f"The number of bootstrap iterations in the boostrapping method, if used: {self.n_bootstrap}")
            print(f"The random seed to use in bootstrapping, if used: {self.seed}")

        if self.reformatted_mdp is True:
            print("Note that the input MDP file has been reformatted by replacing hypens with underscores. The original mdp file has been renamed as *backup.mdp.")  # noqa: E501

    @staticmethod
    def reformat_MDP(mdp_file):
        """
        Reformats the input MDP file so that all hyphens in the parameter names are replaced by underscores.
        If the input MDP file contains hyphens in its parameter names, the class attribue :code:`self.reformatted`
        will be set to :code:`True`. In this case, the new MDP object with reformatted parameter names will be
        written to the original file path of the file, while the original file will be renamed with a
        :code:`_backup` suffix. If the input MDP file is not reformatted, the function sets
        the attribute :code:`self.reformatted_mdp` to :code:`False`.

        Parameters
        ----------
        mdp_file : str
            The file path of the MDP file to be reformatted.

        Returns
        -------
        reformatted : bool
            Whether the file was reformatted.
        """
        params = gmx_parser.MDP(mdp_file)

        odict = OrderedDict([(k.replace('-', '_'), v) for k, v in params.items()])
        params_new = gmx_parser.MDP(None, **odict)
        reformatted = None

        if rank == 0:
            if params_new.keys() == params.keys():
                reformatted = False  # no need to reformat the file
            else:
                reformatted = True
                new_name = mdp_file.split('.mdp')[0] + '_backup.mdp'
                os.rename(mdp_file, new_name)
                params_new.write(mdp_file)

        return reformatted

    def initialize_MDP(self, idx):
        """
        Initializes the MDP object for generating an MDP file for a specific replica based on the MDP template.
        This function should be called only for generating MDP files for the FIRST iteration
        and it has nothing to do with whether the weights are fixed or equilibrating.
        It is assumed that the MDP template has all the parameters shared by all replicas.

        Parameters
        ----------
        idx : int
            Index of the simulation whose MDP parameters need to be initialized.

        Returns
        -------
        MDP : :obj:`.gmx_parser.MDP` obj
            A :obj:`.gmx_parser.MDP` object that can be used to write the MDP file.
        """
        MDP = copy.deepcopy(self.template)
        MDP["nsteps"] = self.nst_sim

        if self.mdp_args is not None:
            for param in self.mdp_args.keys():
                MDP[param] = self.mdp_args[param][idx]

        start_idx = idx * self.s
        for i in self.lambda_types:
            MDP[i] = self.template[i][start_idx:start_idx + self.n_sub]

        if "init_lambda_weights" in self.template:
            MDP["init_lambda_weights"] = self.template["init_lambda_weights"][start_idx:start_idx + self.n_sub]

        return MDP

    def get_ref_dist(self, pullx_file=None):
        """
        Gets the initial COM distance between the pull groups in the input GRO file. Importantly, this distance
        will serve as the reference distance starting from the second iteration. This function initializes the
        attribute :code:`ref_dist` and is only relevant when a distance restraint is applied in the GROMACS pull code.

        Parameters
        ----------
        pullx_file : str, Optional
            The path of the pullx file whose initial value will be used as the reference distance.
            Usually, this should be the path of the pullx file of the first iteration. The default
            is :code:`sim_0/iteration_0/pullx.xvg`.
        """
        if pullx_file is None:
            pullx_file = f"{self.working_dir}/sim_0/iteration_0/pullx.xvg"
        if hasattr(self, 'set_ref_dist'):
            self.ref_dist = []
            for i in range(len(self.set_ref_dist)):
                if self.set_ref_dist[i] is True:
                    # dist = list(extract_dataframe(pullx_file, headers=headers)[f'{i+1}'])[0]
                    dist = np.loadtxt(pullx_file, comments=['#', '@'])[0][1]
                    self.ref_dist.append(dist)

    def update_MDP(self, new_template, sim_idx, iter_idx, states, wl_delta, weights, counts=None):
        """
        Updates the MDP file for a new iteration based on the new MDP template, which is the MDP file
        from the previous iteration. Note that if the weights got equilibrated in the previous iteration,
        the weights will be fixed at these equilibrated values for all the following iterations.

        Parameters
        ----------
        new_template : str
            The new MDP template file, which typically is the MDP file of the previous iteration.
        sim_idx : int
            The index of the simulation whose MDP parameters need to be updated.
        iter_idx : int
            The index of the iteration to be performed later.
        states : list
            A list of last sampled states of all simulaitons in the previous iteration.
        wl_delta : list
            A list of fina Wang-Landau incrementors of all simulations.
        weights : list
            A list of lists final weights of all simulations.
        counts : list, Optional
            A list of lists final counts of all simulations. If the value is :code:`None`,
            then the MDP parameter :code:`init-histogram-counts` won't be specified in the next iteration.
            Note that this parameter is only supported by GROMACS with versions later than 2022.3.

        Return
        ------
        MDP : :obj:`.gmx_parser.MDP` obj
            A :obj:`.gmx_parser.MDP` object that can be used to write the MDP file.
        """
        new_template = gmx_parser.MDP(new_template)  # turn into a gmx_parser.MDP object
        MDP = copy.deepcopy(new_template)
        MDP["tinit"] = self.nst_sim * self.dt * iter_idx
        MDP["nsteps"] = self.nst_sim
        MDP["init_lambda_state"] = (states[sim_idx] - sim_idx * self.s)  # 2nd term is for shifting from the global to local index.  # noqa: E501
        MDP["init_wl_delta"] = wl_delta[sim_idx]
        MDP["init_lambda_weights"] = weights[sim_idx]
        if counts is not None:
            MDP["init_histogram_counts"] = counts[sim_idx]

        if self.equil[sim_idx] == -1:   # the weights haven't been equilibrated
            MDP["init_wl_delta"] = wl_delta[sim_idx]
        else:
            MDP["lmc_stats"] = "no"
            MDP["wl_scale"] = ""
            MDP["wl_ratio"] = ""
            MDP["init_wl_delta"] = ""
            MDP["lmc_weights_equil"] = ""
            MDP["weight_equil_wl_delta"] = ""

        # Here we deal with the distance restraint in the pull code, if any.
        if hasattr(self, 'ref_dist'):
            for i in range(len(self.ref_dist)):
                MDP[f'pull_coord{i+1}_start'] = "no"
                MDP[f'pull_coord{i+1}_init'] = self.ref_dist[i]

        return MDP

    def extract_final_dhdl_info(self, dhdl_files):
        """
        Extracts the last sampled states for all replica simulations.

        Parameters
        ----------
        dhdl_files : list
            A list of file paths to GROMACS DHDL files of different replicas. Note that
            the order of the files should be consistent with the order of the replicas.

        Returns
        -------
        states : list
            A list of the global state indices of the last sampled states of all simulaitons.
        """
        states = []
        print("\nBelow are the final states being visited:")
        for i in range(len(dhdl_files)):
            headers = get_headers(dhdl_files[i])
            state_local = list(extract_dataframe(dhdl_files[i], headers=headers)['Thermodynamic state'])[-1]  # local index of the last state  # noqa: E501
            state_global = state_local + i * self.s  # global index of the last state
            states.append(state_global)  # append the global index
            print(f"  Simulation {i}: Global state {states[i]}")
        print('\n', end='')

        return states

    def extract_final_log_info(self, log_files):
        """
        Extracts the following information for each replica simulation from a given list of GROMACS LOG files:

          - The final Wang-Landau incrementors.
          - The final lists of weights.
          - The final lists of counts.

        Note that the order of the files should be consistent with the order of the replicas.

        Parameters
        ----------
        log_files : list
            A list of file paths to GROMACS LOG files of different replicas.

        Returns
        -------
        wl_delta : list
            A list of final Wang-Landau incrementors of all simulations.
        weights : list
            A list of lists of final weights of all simulations.
        counts : list
            A list of lists of final counts of all simulations.
        """
        wl_delta, weights, counts = [], [], []

        # Find the final Wang-Landau incrementors and weights
        for i in range(len(log_files)):
            if self.verbose:
                print(f'Parsing {log_files[i]} ...')
            result = gmx_parser.parse_log(log_files[i])  # weights, counts, wl_delta, equil_time
            weights.append(result[0][-1])
            counts.append(result[1])
            wl_delta.append(result[2])

            # In Case 3 described in the docstring of parse_log (fixed-weights),
            # result[3] will be 0 but it will never be passed to self.equil[i]
            # because once self.equil[i] is not -1, we stop updating. This way, we can keep
            # the time when the weights get equilibrated all the way.
            if self.equil[i] == -1:
                # At this point self.equil is the equilibration status BEFORE the last iteration
                # while result[3] is the equilibration status AFTER finishing the last iteraiton.
                # For any replicas where weights are still equilibrating (i.e. self.equil[j] == -1)
                # we update its equilibration status.
                self.equil[i] = result[3]

            if self.equilibrated_weights[i] == []:
                if self.equil[i] != -1 and self.equil[i] != 0:
                    # self.equil[i] != -1: uneqilibrated
                    # self.equil[i] != 0: fixed-weight simulation
                    self.equilibrated_weights[i] = result[0][-1]

        return wl_delta, weights, counts

    def identify_swappable_pairs(self, states, state_ranges, dhdl_files, iteration=None):  # noqa: E501
        """
        Identifies swappable pairs. By definition, a pair of simulation is considered swappable only if
        their last sampled states are in the alchemical ranges of both simulations. This is required
        to ensure that the values of involved ΔH and Δg can always be looked up from the DHDL and LOG files.
        This also automatically guarantees that the simulations to be swapped have overlapping state sets.

        Parameters
        ----------
        states : list
            A list of the global state indices of the last sampled states of all simulations. This list can be
            generated by :obj:`.extract_final_dhdl_info`. Notably, the input list should not be
            a list that has been updated/modified by :obj:`get_swapping_pattern`, or the result will be incorrect.
        state_ranges : list of lists
            A list of global state indices for all replicas. The input list can be a list updated by
            :obj:`.get_swapping_pattern`, especially in the case where there is a need to re-identify the
            swappable pairs after an attempted swap is accepted.
        iteration : int
            The iteration number will determine which direction the swaps will be performed. This option is
            only necessary when using the "random_range" proposal scheme.

        Returns
        -------
        swappables : list
            A list of tuples representing the simulations that can be swapped.

        Example
        -------
        Below is an example where the REXEE simulation is composed of four replicas sampling states 0-3, 1-4,
        2-5, and 3-6, respectively. At exchanges, these replicas are respectively at states 2, 3, 3, and 4.
        Therefore, the swappable pairs are [(0, 1), (1, 2), (2, 3)]. If only neighboring swaps are considered,
        the swappable pairs will be [(1, 2), (2, 3)].

            >>> from ensemble_md.replica_exchange_EE import ReplicaExchangeEE as REXEE
            >>> states = [2, 3, 2, 5]
            >>> state_ranges = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
            >>> REXEE.identify_swappable_pairs(states, state_ranges, neighbor_exchange=False)
            [(0, 1), (1, 2), (2, 3)]
            >>> REXEE.identify_swappable_pairs(states, state_ranges, neighbor_exchange=True)
            [(1, 2), (2, 3)]
        """
        n_sim = len(states)
        sim_idx = list(range(n_sim))
        states_for_swap = []
        if self.proposal == 'random_range':
            potential_swappables = []
            for n in np.arange(0, n_sim-1, 1):
                potential_swappables.append([n, n+1])
            swappables, swap_index = [], []
            for swap in potential_swappables:
                index, state = self._deter_swap_index(swap, dhdl_files, self.add_swappables)
                if len(index) == 2:
                    swap_index.append(index)
                    swappables.append(swap)
                    states_for_swap.append(state)
        else:
            all_pairs = list(combinations(sim_idx, 2))

            # First, we identify pairs of replicas with overlapping ranges
            swappables = [i for i in all_pairs if set(state_ranges[i[0]]).intersection(set(state_ranges[i[1]])) != set()]  # noqa: E501

            # Then, from these pairs, we exclude the ones whose the last sampled states are not present in both alchemical ranges  # noqa: E501
            # In this case, U^i_n, U_^j_m, g^i_n, and g_^j_m are unknown and the acceptance cannot be calculated.
            swappables = [i for i in swappables if states[i[0]] in state_ranges[i[1]] and states[i[1]] in state_ranges[i[0]]]  # noqa: E501

            # Expand the definition of swappable pairs when add_swappables is specified
            if self.add_swappables is not None:
                all_paired_states = [[states[p[0]], states[p[1]]] for p in all_pairs]
                for i in all_paired_states:
                    if i in self.add_swappables:
                        pair = all_pairs[all_paired_states.index(i)]
                        if pair not in swappables:
                            swappables.append(pair)

            if self.proposal == 'neighboring':
                swappables = [i for i in swappables if np.abs(i[0] - i[1]) == 1]
            swap_index = []

        return swappables, swap_index, states_for_swap

    @staticmethod
    def propose_swap(swappables):
        """
        Proposes a swap of coordinates between replicas by drawing a pair from the list of swappable pairs.

        Parameters
        ----------
        swappables : list
            A list of tuples representing the simulations that can be swapped.

        Return
        ------
        swap : tuple or an empty list
            A tuple of simulation indices to be swapped. If there is no swappable pair,
            an empty list will be returned.
        """
        try:
            swap = random.choices(swappables, k=1)[0]
        except IndexError:  # no swappable pairs
            swap = []

        return swap

    def get_swapping_pattern(self, dhdl_files, states, iteration=None):
        """
        Generates a list (:code:`swap_pattern`) that represents how the configurations should be swapped in the
        next iteration. The indices of the output list correspond to the simulation/replica indices, and the
        values represent the configuration indices in the corresponding simulation/replica. For example, if the
        swapping pattern is :code:`[0, 2, 1, 3]`, it means that in the next iteration, replicas 0, 1, 2, 3 should
        sample configurations 0, 2, 1, 3, respectively, where configurations 0, 1, 2, 3 here are defined as whatever
        configurations are in replicas 0, 1, 2, 3 in the CURRENT iteration (not iteration 0), respectively.

        Notably, when this function is called (e.g., once every iteration in a REXEE simulation), the output
        list :code:`swap_pattern` is always initialized as :code:`[0, 1, 2, 3, ...]` and gets updated once every
        attempted swap. This is different from the attribute :code:`configs`, which is only initialized at the
        very beginning of the entire REXEE simulation (iteration 0), though :code:`configs` also gets updated with
        :code:`swap_pattern` once every attempted swap in this function.

        Parameters
        ----------
        dhdl_files : list
            A list of paths to the DHDL files. The indicies in the DHDL filenames should be in an ascending order, e.g.
            :code:`[dhdl_0.xvg, dhdl_1.xvg, ..., dhdl_N.xvg]`.
        states : list
            A list of last sampled states (in global indices) of ALL simulations. :code:`states[i]=j` means that
            the configuration in replica :code:`i` is at state :code:`j` at the time when the exchange is performed.
            This list can be generated :obj:`.extract_final_dhdl_info`.
        iteration : int
            The iteration number will determine which direction the swaps will be performed. This option is only
            necessary when using the "random_range" proposal scheme.

        Returns
        -------
        swap_pattern : list
            A list showing the configuration of replicas after swapping.
        swap_list : list
            A list of tuples showing the accepted swaps.
        """
        swap_list = []
        if self.proposal == 'random_range':
            swap_index = []
            n_ex = int(np.floor(self.n_sim / 2))  # This is the maximum, not necessarily the number that will always be reached.  # noqa
            n_ex_FR = 0
        elif self.proposal == 'exhaustive':
            n_ex = int(np.floor(self.n_sim / 2))  # This is the maximum, not necessarily the number that will always be reached.  # noqa
            n_ex_exhaustive = 0    # The actual number of swaps atttempted.
        else:
            n_ex = 1  # single swap or neighboring swap

        """
        # multiple swaps
        if self.n_ex == 'N^3':
            n_ex = self.n_tot ** 3
        else:
            n_ex = self.n_ex
        """

        shifts = list(self.s * np.arange(self.n_sim))
        swap_pattern = list(range(self.n_sim))   # Can be regarded as the indices of DHDL files/configurations
        state_ranges = copy.deepcopy(self.state_ranges)
        # states_copy = copy.deepcopy(states)  # only for re-identifying swappable pairs given updated state_ranges --> was needed for the multiple exchange proposal scheme  # noqa: E501
        swappables, swap_index, states_for_swap = self.identify_swappable_pairs(states, state_ranges, dhdl_files, iteration)  # noqa: E501
        all_swappables = swappables.copy()

        # Note that if there is only 1 swappable pair, then it will still be the only swappable pair
        # after an attempted swap is accepted. Therefore, there is no need to perform multiple swaps or re-identify
        # the new set of swappable pairs. In this case, we set n_ex back to 1.
        if len(swappables) == 1:
            if n_ex > 1:
                n_ex = 1  # n_ex is set back to 1 since there is only 1 swappable pair.

        states_modified = states
        swap_index_accept = []
        for i in range(n_ex):
            # Update the list of swappable pairs starting from the 2nd attempted swap for the exhaustive swap method.
            if (self.proposal == 'exhaustive' or self.proposal == 'random_range') and i >= 1:  # noqa: E501
                # Note that this should be done regardless of the acceptance/rejection of the previously drawn pairs.
                # Also note that at this point, swap is still the last attempted swap.
                swappables = [i for i in swappables if set(i).intersection(set(swap)) == set()]  # noqa: F821
                print(f'\nRemaining swappable pairs: {swappables}')

            if len(swappables) == 0 and (self.proposal == 'exhaustive' or self.proposal == 'random_range'):  # noqa: E501
                # This should only happen when the method of exhaustive swaps is used.
                if i == 0:
                    self.n_empty_swappable += 1
                    print('No swap is proposed because there is no swappable pair at all.')
                break
            else:
                if self.proposal == 'exhaustive':
                    n_ex_exhaustive += 1
                if self.proposal == 'random_range':
                    n_ex_FR += 1

                swap = ReplicaExchangeEE.propose_swap(swappables)

                print(f'\nProposed swap: {swap}')
                if swap == []:  # the same as len(swappables) == 0, self.proposal must not be exhaustive if this line is reached.  # noqa: E501
                    self.n_empty_swappable += 1
                    print('No swap is proposed because there is no swappable pair at all.')
                    break  # no need to re-identify swappable pairs and draw new samples
                else:
                    self.n_swap_attempts += 1
                    if self.verbose is True and self.proposal != 'exhaustive' and self.proposal != 'random_range':  # noqa: E501
                        print(f'A swap ({i + 1}/{n_ex}) is proposed between the configurations of Simulation {swap[0]} (state {states[swap[0]]}) and Simulation {swap[1]} (state {states[swap[1]]}) ...')  # noqa: E501

                    if self.modify_coords_fn is not None:
                        swap_bool = True  # always accept the move
                    else:
                        # Calculate the acceptance ratio and decide whether to accept the swap.
                        prob_acc = self.calc_prob_acc(swap, dhdl_files, states, shifts)
                        swap_bool = self.accept_or_reject(prob_acc)

                    # Theoretically, in an REXEE simulation, we could either choose to swap configurations (via
                    # swapping GRO files) or replicas (via swapping MDP files). In ensemble_md package, we chose the
                    # former when implementing the REXEE algorithm. Specifically, in the CLI `run_REXEE`,
                    # `swap_pattern` is used to swap the GRO files. Therefore, when an attempted swap is accetped and
                    # `swap_pattern` is updated, we also need to update the variables `shifts`, `weights`,
                    # `dhdl_files`, `state_ranges`, `self.configs` but not anything else. Otherwise, incorrect results
                    # will be produced. To better understand this, one can refer to our unit test for
                    # get_swapping_pattern and calc_prob_acc, set checkpoints and examine why the variables
                    # should/should not be updated.

                    if swap_bool is True:
                        swap_list.append(swap)
                        if self.proposal == "random_range":
                            for p, p_swap in enumerate(all_swappables):
                                if p_swap == swap:
                                    break
                            swap_index_accept.append(swap_index[p])
                            states_modified[swap[0]] = states_for_swap[p][0]
                            states_modified[swap[1]] = states_for_swap[p][1]
                        else:
                            swap_index_accept.append([-1, -1])
                        # Determine which
                        # The assignments need to be done at the same time in just one line.
                        # states[swap[0]], states[swap[1]] = states[swap[1]], states[swap[0]]
                        # weights[swap[0]], weights[swap[1]] = weights[swap[1]], weights[swap[0]]
                        shifts[swap[0]], shifts[swap[1]] = shifts[swap[1]], shifts[swap[0]]
                        dhdl_files[swap[0]], dhdl_files[swap[1]] = dhdl_files[swap[1]], dhdl_files[swap[0]]
                        swap_pattern[swap[0]], swap_pattern[swap[1]] = swap_pattern[swap[1]], swap_pattern[swap[0]]
                        state_ranges[swap[0]], state_ranges[swap[1]] = state_ranges[swap[1]], state_ranges[swap[0]]
                        self.configs[swap[0]], self.configs[swap[1]] = self.configs[swap[1]], self.configs[swap[0]]

                        """
                        if n_ex > 1 and self.proposal == 'multiple':  # must be multiple swaps
                            # After state_ranges have been updated, we re-identify the swappable pairs.
                            # Notably, states_copy (instead of states) should be used. (They could be different.)
                            swappables = ReplicaExchangeEE.identify_swappable_pairs(states_copy, state_ranges, self.proposal == 'neighboring', self.add_swappables)  # noqa: E501
                            print(f"  New swappable pairs: {swappables}")
                        """
                    else:
                        # In this case, there is no need to update the swappables
                        pass

                print(f'  Current list of configurations: {self.configs}')

        if self.verbose is False:
            print(f'\n{n_ex} swap(s) have been proposed.')
        print(f'\nThe finally adopted swap pattern: {swap_pattern}')
        print(f'The list of configurations sampled in each replica in the next iteration: {self.configs}')

        # Update the replica-space trajectories
        for i in range(self.n_sim):
            self.rep_trajs[i].append(self.configs.index(i))

        return swap_pattern, swap_list, swap_index_accept, states_modified

    def calc_prob_acc(self, swap, dhdl_files, states, shifts):
        """
        Calculates the acceptance ratio for swapping simulations.

        Parameters
        ----------
        swap : tuple
            A tuple of indices corresponding to the simulations to be swapped.
        dhdl_files : list
            A list of DHDL files, e.g. :code:`dhdl_files = ['dhdl_2.xvg', 'dhdl_1.xvg', 'dhdl_0.xvg', 'dhdl_3.xvg']`
            means that configurations 2, 1, 0, and 3 are now in replicas 0, 1, 2, 3. This can happen in multiple swaps
            when a previous swap between configurations 0 and 2 has just been accepted. Otherwise, the list of
            filenames should always be in the ascending order, e.g., :code:`['dhdl_0.xvg', 'dhdl_1.xvg', 'dhdl_2.xvg',
            dhdl_3.xvg]`.
        states : list
            A list of last sampled states (in global indices) in the DHDL files corresponding to configurations 0, 1,
            2, ... (e.g., :code:`dhdl_0.xvg`, :code:`dhdl_1.xvg`, :code:`dhdl_2.xvg`, ...)
            This list can be generated by :obj:`.extract_final_dhdl_info`.
        shifts : list
            A list of state shifts for converting global state indices to the local ones. Specifically, :code:`states`
            substracted by :code:`shifts` should be the local state indices of the last sampled states
            in :code:`dhdl_files[0]`, :code:`dhdl_files[1]`, ... (which importantly, are not necessarily
            :code:`dhdl_0.xvg`, :code:`dhdl_1.xvg`, ...). If multiple swaps are not used, then
            this should always be :code:`list(REXEE.s * np.arange(REXEE.n_sim))`.

        Returns
        -------
        prob_acc : float
            The acceptance ratio.
        """
        if self.verbose is True:
            print("  Proposing a move from (x^i_m, x^j_n) to (x^i_n, x^j_m) ...")
        f0, f1 = dhdl_files[swap[0]], dhdl_files[swap[1]]
        h0, h1 = get_headers(f0), get_headers(f1)
        data_0, data_1 = (
            extract_dataframe(f0, headers=h0).iloc[-1],
            extract_dataframe(f1, headers=h1).iloc[-1],
        )

        # \Delta H to all states at the last time frame
        # Notably, the can be regarded as H for each state since the reference state will have a value of 0 anyway.
        dhdl_0 = data_0[-self.n_sub:]
        dhdl_1 = data_1[-self.n_sub:]

        # Old local index, which will only be used in "metropolis"
        old_state_0 = states[swap[0]] - shifts[swap[0]]
        old_state_1 = states[swap[1]] - shifts[swap[1]]

        # New local index. Note that states are global indices, so we shift them back to the local ones
        new_state_0 = states[swap[1]] - shifts[swap[0]]  # new state index (local index in simulation swap[0])
        new_state_1 = states[swap[0]] - shifts[swap[1]]  # new state index (local index in simulation swap[1])

        dU_0 = (dhdl_0[new_state_0] - dhdl_0[old_state_0]) / self.kT  # U^{i}_{n} - U^{i}_{m}, i.e. \Delta U (kT) to the new state  # noqa: E501
        dU_1 = (dhdl_1[new_state_1] - dhdl_1[old_state_1]) / self.kT  # U^{j}_{m} - U^{j}_{n}, i.e. \Delta U (kT) to the new state  # noqa: E501
        dU = dU_0 + dU_1
        if self.verbose is True:
            print(
                f"  U^i_n - U^i_m = {dU_0:.2f} kT, U^j_m - U^j_n = {dU_1:.2f} kT, Total dU: {dU:.2f} kT"
            )

        prob_acc = min(1, np.exp(-dU))

        return prob_acc

    def accept_or_reject(self, prob_acc):
        """
        Returns a boolean variable indicating whether the proposed swap should be acceepted or not given
        the acceptance ratio.

        Parameters
        ----------
        prob_acc : float
            The acceptance ratio.

        Returns
        -------
        swap_bool : bool
            A boolean variable indicating whether the swap should be accepted.
        """
        if prob_acc == 0:
            swap_bool = False
            self.n_rejected += 1
            if self.verbose is True:
                print("  Swap rejected! ", end="", flush=True)
        else:
            rand = random.random()
            if self.verbose is True:
                print(
                    f"  Acceptance rate: {prob_acc:.3f} / Random number drawn: {rand:.3f}"
                )
            if rand < prob_acc:
                swap_bool = True
                if self.verbose is True:
                    print("  Swap accepted! ")
            else:
                swap_bool = False
                self.n_rejected += 1
                if self.verbose is True:
                    print("  Swap rejected! ")
        return swap_bool

    def _deter_swap_index(self, swap, dhdl_files, add_swappables):
        """
        Determine which frames in the trajectory are sampling compatible lambda states to engage in a swap

        Parameters
        ----------
        swap : list
        The simulations for which the swap is occuring
        dhdl_files : list
            A list of paths to the DHDL files. The indicies in the DHDL filenames should be in an ascending order, e.g.
            :code:`[dhdl_0.xvg, dhdl_1.xvg, ..., dhdl_N.xvg]`.
        add_swappables: list
            A list of lists that additionally consider states (in global indices) that can be swapped.
            For example, :code:`add_swappables=[[4, 5], [14, 15]]` means that if a replica samples state 4,
            it can be swapped with another replica that samples state 5 and vice versa. The same logic applies
            to states 14 and 15. This parameter is only relevant to MT-REXEE simulations.

        Returns
        -------
        swap_bool : bool
            A boolean variable indicating whether the swap should be accepted.
        """
        # Determine possible lambda states that swap partner i can be in to allow a swap
        swappable_global_states, swap_state = [], []
        for swappable in add_swappables:
            A, B = swappable
            if (A >= (swap[0] * self.s) and A < ((swap[0] + 1) * self.s) and B >= (swap[1] * self.s) and B < ((swap[1] + 1) * self.s)) or (B >= (swap[0] * self.s) and B < ((swap[0] + 1) * self.s) and A >= (swap[1] * self.s) and A < ((swap[1] + 1) * self.s)):  # noqa: E501
                swappable_global_states.append(A)
                swappable_global_states.append(B)
        convert_to_frames = int(self.template["nstxout"]/self.template["nstdhdl"])
        swap_index = []
        for i in range(2):
            n = swap[i]
            # Determine all frames for which swap partner i is in those states
            potential_swap_index, state_global_list = [], []
            headers = get_headers(dhdl_files[n])
            state_local = list(extract_dataframe(dhdl_files[n], headers=headers)['Thermodynamic state'])  # local index for all states  # noqa: E501
            for s, state in enumerate(state_local[::convert_to_frames]):
                state_global = state + (n * self.s)  # global index of the last state
                if state_global in swappable_global_states:
                    potential_swap_index.append(s)
                    state_global_list.append(state_global)
            # Select a random frame which is in the last 50% of the trajectory to have the swap occur
            potential_swap_index = np.array(potential_swap_index)
            potential_swap_index = potential_swap_index[potential_swap_index > (len(state_local)/(2*convert_to_frames))]  # noqa: E501
            if len(potential_swap_index) != 0:
                index = np.random.choice(potential_swap_index)
                swap_state.append(state_global_list[np.where(potential_swap_index == index)[0][0]])
                swap_index.append(index)
            else:
                break
        return swap_index, swap_state

    def get_averaged_weights(self, log_files):
        """
        For each replica, calculates the averaged weights (and the associated error) from the time series
        of the weights since the previous update of the Wang-Landau incrementor. This is only relevant
        for weight-updating REXEE simulations.

        Parameters
        ----------
        log_files : list
            A list of file paths to GROMACS LOG files of different replicas.

        Returns
        -------
        weights_avg : list
            A list of lists of weights averaged since the last update of the Wang-Landau
            incrementor. The length of the list should be the number of replicas.
        weights_err : list
            A list of lists of errors corresponding to the values in :code:`weights_avg`.
        """
        for i in range(self.n_sim):
            weights, _, wl_delta, _ = gmx_parser.parse_log(log_files[i])
            if self.current_wl_delta[i] == wl_delta:
                self.updating_weights[i] += weights  # expand the list
            else:
                self.current_wl_delta[i] = wl_delta
                self.updating_weights[i] = weights

        # shape of self.updating_weights is (n_sim, n_points, n_states), but n_points can be different
        # for different replicas, which will error out np.mean(self.updating_weights, axis=1)
        weights_avg = [np.mean(self.updating_weights[i], axis=0).tolist() for i in range(self.n_sim)]
        weights_err = []
        for i in range(self.n_sim):
            if len(self.updating_weights[i]) == 1:  # this would lead to a RunTime Warning and nan
                weights_err.append([0] * self.n_sub)  # in `weighted_mean``, a simple average will be returned.
            else:
                weights_err.append(np.std(self.updating_weights[i], axis=0, ddof=1).tolist())

        return weights_avg, weights_err

    def weight_correction(self, weights, counts):
        """
        Adjusts the lambda weights based on the histogram counts by using the following equation:
        :math:`g_k' = g_k + ln(N_{k-1}/N_k)`, where :math:`g_k` and :math:`g_k'`
        are the lambda weight before and after the correction, respectively.
        Notably, in any of the following situations, we don't do any correction.

        - Either :math:`N_{k-1}` or :math:`N_k` is :math:`0`.
        - Either :math:`N_{k-1}` or :math:`N_k` is smaller than the histogram cutoff specified by :code:`N_cutoff`
          in the input YAML file.

        Parameters
        ----------
        weights : list
            A list of lists of weights (of ALL simulations) to be corrected. The i-th element corresponds to
            the list of weights of the i-th replica.
        counts : list
            A list of lists of counts (of ALL simulations).

        Return
        ------
        weights : list
            An updated list of lists of corected weights.
        """
        skip_correction = False
        for i in range(len(weights)):  # loop over the replicas
            if self.verbose is True:
                print(f"  Counts of rep {i}:\t\t{counts[i]}")
                print(f'  Original weights of rep {i}:\t{[float(f"{k:.3f}") for k in weights[i]]}')

            for j in range(1, len(weights[i])):  # loop over the alchemical states
                if counts[i][j - 1] != 0 and counts[i][j - 1] != 0:
                    if np.min([counts[i][j - 1], counts[i][j]]) > self.N_cutoff:
                        weights[i][j] += np.log(counts[i][j - 1] / counts[i][j])
                    else:
                        skip_correction = True
                        print('Weight correction was deactivated because neither N_{k-1} or N_k is larger than the histogram cutoff.')  # noqa: E501

            if self.verbose is True and skip_correction is False:
                print(f'  Corrected weights of rep {i}:\t{[float(f"{k:.3f}") for k in weights[i]]}\n')

        if self.verbose is False:
            print(' Done')

        return weights

    def histogram_correction(self, hist, print_values=True):
        """
        Adjusts the histogram counts. For example, if replicas A and B both sample states 1 and 2 and have
        histogram counts :math:`N^A_1`, :math:`N^A_2`, :math:`N^B_1`, and :math:`N^B_2`, the corrected histogram
        counts for states 1 and 2 for BOTH replicas will be adjusted according to the following equation:
        :math:`N_1'/N_2'=((N_1^A N_1^B)/(N_2^A N_2^B))^{1/2}`. Namely, the ratio of the corrected histogram
        counts for adjacent states is the geometric mean of the ratio of the original histogram counts
        for the same states. Note that if any histogram count is 0, histogram correction will not be performed
        and the original histogram counts will be returned.

        Parameters
        ----------
        hist : list
            A list of lists of histogram counts of ALL simulations. The i-th element corresponds to
            the list of histogram counts of the i-th replica.
        print_values : bool, Optional
            Whether to print the histogram counts for each replica before and after histogram correction.
            The default is :code:`True`.

        Returns
        -------
        hist_modified : list
            A list of lists of modified histogram counts of ALL simulations.
        """
        # (1) Print the original histogram counts
        if print_values is True:
            print('  Original histogram counts:')
            for i in range(len(hist)):
                print(f'    Rep {i}: {hist[i]}')

        # (2) Calculate adjacent weight differences and g_vec
        N_ratio_vec = []  # N_{k-1}/N_k for the whole range
        with warnings.catch_warnings():  # Suppress the specific warning here
            warnings.simplefilter("ignore", category=RuntimeWarning)
            N_ratio_adjacent = [list(np.array(hist[i][1:]) / np.array(hist[i][:-1])) for i in range(len(hist))]

        for i in range(self.n_tot - 1):
            N_ratio_list = []
            for j in range(len(self.state_ranges)):
                if i in self.state_ranges[j] and i + 1 in self.state_ranges[j]:
                    idx = self.state_ranges[j].index(i)
                    N_ratio_list.append(N_ratio_adjacent[j][idx])
            N_ratio_vec.append(np.prod(N_ratio_list) ** (1 / len(N_ratio_list)))  # geometric mean
        N_ratio_vec.insert(0, hist[0][0])

        # (3) Check if the histogram counts are 0 for some states, if so, the histogram correction will be skipped.
        # Zero histogram counts can happen when the sampling is poor or the WL incrementor just got updated
        contains_nan = any(np.isnan(value) for sublist in N_ratio_adjacent for value in sublist)  # can be caused by 0/0  # noqa: E501
        contains_inf = any(np.isinf(value) for sublist in N_ratio_adjacent for value in sublist)  # can be caused by x/0, where x is a finite number  # noqa: E501
        skip_hist_correction = contains_nan or contains_inf
        if skip_hist_correction:
            print('\n  Histogram correction is skipped because the histogram counts are 0 for some states.')

        # (4) Perform histogram correction if it is not skipped
        if skip_hist_correction is False:
            print('\n  Performing histogram correction ...')
            # When skip_hist_correction is True, previous lines for calculating N_ratio_vec or N_ratio_list will
            # still not error out so it's fine to not add the conditional statement like here, since we will
            # have hist_modified = hist at the end anyway. However, if skip_hist_correction, things like
            # int(np.nan) will lead to an error, so we put an if condition here.
            N_vec = np.array([int(np.ceil(np.prod(N_ratio_vec[:(i + 1)]))) for i in range(len(N_ratio_vec))])

        if skip_hist_correction is False:
            hist_modified = [list(N_vec[self.state_ranges[i]]) for i in range(self.n_sim)]
        else:
            hist_modified = hist  # the original input histogram

        # (5) Print the modified histogram counts
        if print_values is True:
            print('\n  Modified histogram counts:')
            for i in range(len(hist_modified)):
                print(f'    Rep {i}: {hist_modified[i]}')

        return hist_modified

    def combine_weights(self, weights, weights_err=None, print_values=True):
        """
        Combines alchemical weights across multiple replicas. Note that if
        :code:`weights_err` is provided, inverse-variance weighting will be used.
        Care must be taken since inverse-variance weighting can lead to slower
        convergence if the provided errors are not accurate. (See :ref:`doc_w_schemes` for more details.)

        Parameters
        ----------
        weights : list
            A list of lists of alchemical weights of ALL simulations. The i-th element corresponds to
            the list of weights of the i-th replica.
        weights_err : list, Optional
            A list of lists of errors corresponding to the values in :code:`weights`. The default is :code:`None`.
        print_values : bool, Optional
            Whether to print the weights for each replica before and
            after weight combination for each replica. The default is :code:`True`.

        Returns
        -------
        weights_modified : list
            A list of modified alchemical weights of ALL simulations.
        g_vec : numpy.ndarray
            An array of alchemical weights of the whole range of states.
        """
        # (1) Print the original weights
        if print_values is True:
            w = np.round(weights, decimals=3).tolist()  # just for printing
            print('  Original weights:')
            for i in range(len(w)):
                print(f'    Rep {i}: {w[i]}')

        # (2) Calculate adjacent weight differences and g_vec
        dg_vec = []  # alchemical weight differences for the whole range
        dg_adjacent = [list(np.diff(weights[i])) for i in range(len(weights))]

        if weights_err is not None:
            dg_adjacent_err = [[np.sqrt(weights_err[i][j] ** 2 + weights_err[i][j + 1] ** 2) for j in range(len(weights_err[i]) - 1)] for i in range(len(weights_err))]  # noqa: E501

        for i in range(self.n_tot - 1):
            dg_list, dg_err_list = [], []
            for j in range(len(self.state_ranges)):
                if i in self.state_ranges[j] and i + 1 in self.state_ranges[j]:
                    idx = self.state_ranges[j].index(i)
                    dg_list.append(dg_adjacent[j][idx])
                    if weights_err is not None:
                        dg_err_list.append(dg_adjacent_err[j][idx])
            if weights_err is None:
                dg_vec.append(np.mean(dg_list))
            else:
                dg_vec.append(utils.weighted_mean(dg_list, dg_err_list)[0])

        dg_vec.insert(0, 0)
        nan_loc = [i for i, x in enumerate(dg_vec) if np.isnan(x)]
        if len(nan_loc) != 0:
            g_vec = np.zeros(len(dg_vec))
            for i in range(1, len(dg_vec)):
                if i in nan_loc:
                    continue
                else:
                    g_vec[i] = g_vec[i-1] + dg_vec[i]
        else:
            g_vec = np.array([sum(dg_vec[:(i + 1)]) for i in range(len(dg_vec))])

        # (3) Determine the vector of alchemical weights for each replica
        weights_modified = np.zeros_like(weights)
        for i in range(self.n_sim):
            if self.equil[i] == -1:  # unequilibrated
                weights_modified[i] = list(g_vec[i * self.s: i * self.s + self.n_sub] - g_vec[i * self.s: i * self.s + self.n_sub][0])  # noqa: E501
            else:
                weights_modified[i] = self.equilibrated_weights[i]

        # (4) Print the modified weights
        if print_values is True:
            w = np.round(weights_modified, decimals=3).tolist()  # just for printing
            print('\n  Modified weights:')
            for i in range(len(w)):
                print(f'    Rep {i}: {w[i]}')

        if self.verbose is False:
            print(' DONE')
            print(f'The alchemical weights of all states: \n  {list(np.round(g_vec, decimals=3))}')
        else:
            print(f'\n  The alchemical weights of all states: \n  {list(np.round(g_vec, decimals=3))}')

        return weights_modified, g_vec

    def _run_grompp(self, n, swap_pattern):
        """
        Prepares TPR files for the simulation ensemble using the GROMACS :code:`grompp` command.

        Parameters
        ----------
        n : int
            The iteration index (starting from 0).
        swap_pattern : list
            A list generated by :obj:`.get_swapping_pattern`. It represents how the replicas should be swapped.
        """
        args_list = []
        for i in range(self.n_sim):
            arguments = [self.gmx_executable, 'grompp']

            # Input files
            mdp = f"{self.working_dir}/sim_{i}/iteration_{n}/{self.mdp.split('/')[-1]}"
            if n == 0:
                if isinstance(self.gro, list):
                    gro = f"{self.gro[i]}"
                else:
                    gro = f"{self.gro}"
            else:
                gro = f"{self.working_dir}/sim_{swap_pattern[i]}/iteration_{n-1}/confout.gro"  # This effectively swap out GRO files  # noqa: E501

            if isinstance(self.top, list):
                top = f"{self.top[i]}"
            else:
                top = f"{self.top}"

            # Add input file arguments
            arguments.extend(["-f", mdp, "-c", gro, "-p", top])

            # Add output file arguments
            arguments.extend([
                "-o", f"{self.working_dir}/sim_{i}/iteration_{n}/sys_EE.tpr",
                "-po", f"{self.working_dir}/sim_{i}/iteration_{n}/mdout.mdp"
            ])

            # Add additional arguments if any
            if self.grompp_args is not None:
                # Turn the dictionary into a list with the keys alternating with values
                add_args = [elem for pair in self.grompp_args.items() for elem in pair]
                arguments.extend(add_args)

            args_list.append(arguments)

        # Run the GROMACS grompp commands in parallel
        returncode = None  # Initialize as None for all ranks (necessary for the case when -np > n_sim, which is rare)
        if rank == 0:
            print('Generating TPR files ...')
        if rank < self.n_sim:
            returncode, stdout, stderr = utils.run_gmx_cmd(args_list[rank])
            if returncode != 0:
                print(f'Error on rank {rank} (return code: {returncode}):\n{stderr}')

        # gather return codes at rank 0
        code_list = comm.gather(returncode, root=0)

        if rank == 0:
            # Filter out None values which represent ranks that did not execute the command
            code_list = [code for code in code_list if code is not None]
            if code_list != [0] * self.n_sim:
                MPI.COMM_WORLD.Abort(1)   # Doesn't matter what non-zero returncode we put here as the code from GROMACS will be printed before this point anyway.  # noqa: E501

    def _run_mdrun(self, n):
        """
        Executes GROMACS mdrun commands in parallel for a REXEE simulation.

        Parameters
        ----------
        n : int
            The iteration index (starting from 0).
        """
        # We will change the working directory so the mdrun command should be the same for all replicas.
        arguments = [self.gmx_executable, 'mdrun']

        # Add input file arguments
        arguments.extend(['-s', 'sys_EE.tpr'])

        if self.runtime_args is not None:
            # Turn the dictionary into a list with the keys alternating with values
            add_args = [elem for pair in self.runtime_args.items() for elem in pair]
            arguments.extend(add_args)

        # Run the GROMACS mdrun commands in parallel
        returncode = None  # Initialize as None for all ranks (necessary for the case when -np > n_sim, which is rare)
        if rank == 0:
            print('Running EXE simulations ...')
        if rank < self.n_sim:
            cwd = os.getcwd()
            os.chdir(f'{self.working_dir}/sim_{rank}/iteration_{n}')
            returncode, stdout, stderr = utils.run_gmx_cmd(arguments)
            if returncode != 0:
                print(f'Error on rank {rank} (return code: {returncode}):\n{stderr}')
            if self.rm_cpt is True:
                # if the simulation went wrong, there would be no checkpoint file
                try:
                    os.remove('state.cpt')
                except Exception:
                    print('\n--------------------------------------------------------------------------\n')
                    MPI.COMM_WORLD.Abort(1)
            os.chdir(cwd)

        # gather return codes at rank 0
        code_list = comm.gather(returncode, root=0)

        if rank == 0:
            # Filter out None values which represent ranks that did not execute the command
            code_list = [code for code in code_list if code is not None]
            if code_list != [0] * self.n_sim:
                MPI.COMM_WORLD.Abort(1)   # Doesn't matter what non-zero returncode we put here as the code from GROMACS will be printed before this point anyway.  # noqa: E501

    def run_REXEE(self, n, swap_pattern=None):
        """
        Performs one iteration of a REXEE simulation, which includes generating the
        TPR files using the GROMACS :code:`grompp` command and running the expanded ensemble simulations
        in parallel using the GROMACS :code:`mdrun` command. The GROMACS commands are launched as subprocesses.
        The function assumes that the GROMACS executable is available.

        Parameters
        ----------
        n : int
            The iteration index (starting from 0).
        swap_pattern : list, Optional
            A list generated by :obj:`.get_swapping_pattern`. It represents how the replicas should be swapped.
            This parameter is not needed only if :code:`n` is 0. The default is :code:`None`.
        """
        if rank == 0:
            iter_str = f'\nIteration {n}: {self.dt * self.nst_sim * n: .1f} - {self.dt * self.nst_sim * (n + 1): .1f} ps'  # noqa: E501
            print(iter_str + '\n' + '=' * (len(iter_str) - 1))

        # 1st synchronizing point for all MPI processes: To make sure ranks other than 0 will not start executing
        # _run_grompp earlier and mess up the order of printing.
        comm.barrier()

        # Generating all required TPR files simultaneously, then run all simulations simultaneously.
        # No synchronizing point is needed between _run_grompp and _run_mdrun, since once rank i finishes _run_grompp,
        # it should run _run_mdrun in the same working directory, so there won't be any I/O error.
        self._run_grompp(n, swap_pattern)
        self._run_mdrun(n)

        # 2nd synchronizaing point for all MPI processes: To make sure no rank will start getting to the next
        # iteration earlier than the others. For example, if rank 0 finishes the mdrun command earlier, we don't
        # want it to start parsing the dhdl file (in the if condition of if rank == 0) of simulation 3 being run by
        # rank 3 that has not been generated, which will lead to an I/O error.
        comm.barrier()

    def default_coords_fn(self, molA_file_name, molB_file_name, swap_index):
        """
        Swaps coordinates between two GRO files.

        Parameters
        ----------
        molA_file_name : str
            GRO file name for the moleucle to be swapped.
        molB_file_name : str
            GRO file name for the other moleucle to be swapped.
        """
        # Determine name for transformed residue
        molA_dir = molA_file_name.rsplit('/', 1)[0] + '/'
        molB_dir = molB_file_name.rsplit('/', 1)[0] + '/'

        # Load trajectory trr for higher precison coordinates
        molA = md.load_trr(f'{molA_dir}/traj.trr', top=molA_file_name).slice(swap_index[0])  # Load last frame of trr trajectory  # noqa: E501
        molB = md.load_trr(f'{molB_dir}/traj.trr', top=molB_file_name).slice(swap_index[1])

        # Load the coordinate swapping map
        connection_map = pd.read_csv('residue_connect.csv')
        swap_map = pd.read_csv('residue_swap_map.csv')
        atom_mapping = pd.read_csv('atom_name_mapping.csv')

        # Step 1: Read the GRO input coordinate files and open temporary Output files
        molA_file = open(molA_file_name, 'r').readlines()  # open input file
        molB_new_file_name = 'B_hybrid_swap.gro'
        molB_new = open(molB_new_file_name, 'w')
        molB_file = open(molB_file_name, 'r').readlines()  # open input file
        molA_new_file_name = 'A_hybrid_swap.gro'
        molA_new = open(molA_new_file_name, 'w')

        # Step 2: Determine atoms for alignment and swapping
        residue_options = swap_map['Swap A'].to_list() + swap_map['Swap B'].to_list()
        nameA = coordinate_swap.identify_res(molA.topology, residue_options)
        nameB = coordinate_swap.identify_res(molB.topology, residue_options)
        df_atom_swap = coordinate_swap.extract_missing(nameA, nameB, swap_map)

        # Step 3: Fix break if present for solvated systems only
        if len(molA.topology.select('water')) != 0:
            A_dimensions = coordinate_swap.get_dimensions(molA_file)
            B_dimensions = coordinate_swap.get_dimensions(molB_file)
            molA = coordinate_swap.fix_break(molA, nameA, A_dimensions, connection_map[connection_map['Resname'] == nameA])  # noqa: E501
            molB = coordinate_swap.fix_break(molB, nameB, B_dimensions, connection_map[connection_map['Resname'] == nameB])  # noqa: E501

        # Step 4: Determine coordinates of atoms which need to be reconstructed as we swap coordinates between molecules  # noqa: E501
        df_atom_swap = coordinate_swap.get_miss_coord(molB, molA, nameB, nameA, df_atom_swap, 'A2B', swap_map[(swap_map['Swap A'] == nameB) & (swap_map['Swap B'] == nameA)])  # noqa: E501
        df_atom_swap = coordinate_swap.get_miss_coord(molA, molB, nameA, nameB, df_atom_swap, 'B2A', swap_map[(swap_map['Swap A'] == nameA) & (swap_map['Swap B'] == nameB)])  # noqa: E501

        # Step 5: Parse Current file to ensure atoms are added in the correct order
        atom_order_A = gmx_parser.deter_atom_order(molA_file, nameA)
        atom_order_B = gmx_parser.deter_atom_order(molB_file, nameB)

        # Step 6: Write the new file
        # Reprint preamble text
        line_start = coordinate_swap.print_preamble(molA_file, molB_new, len(df_atom_swap[df_atom_swap['Swap'] == 'A2B']), len(df_atom_swap[df_atom_swap['Swap'] == 'B2A']))  # noqa: E501

        # Print up until we reach the residue to modify
        line_restart, atom_num_restart = coordinate_swap.write_unmodified(line_start, molA_file, molB_new, nameA, 1, line_start, copy.deepcopy(molA.xyz[0]))  # noqa: E501

        # Print new coordinates to file for molB
        line_restart, atom_num_restart = coordinate_swap.write_modified(df_atom_swap, 'A2B', line_start, molA_file, molB_new, atom_num_restart, nameA, nameB, copy.deepcopy(molA.xyz[0]), atom_mapping, atom_order_B, atom_order_A)  # noqa: E501

        if line_restart is not None:
            # Print rest of file after modified residue
            coordinate_swap.write_unmodified(line_restart, molA_file, molB_new, nameA, atom_num_restart, line_start, copy.deepcopy(molA.xyz[0]))  # noqa: E501

        # Print box size
        molB_new.write(molA_file[-1])
        molB_new.close()

        # Print new coordinates to file
        # Reprint preamble text
        line_start = coordinate_swap.print_preamble(molB_file, molA_new, len(df_atom_swap[df_atom_swap['Swap'] == 'B2A']), len(df_atom_swap[df_atom_swap['Swap'] == 'A2B']))  # noqa: E501

        # Print up until we reach the residue to modify
        line_restart, atom_num_restart = coordinate_swap.write_unmodified(line_start, molB_file, molA_new, nameB, 1, line_start, copy.deepcopy(molB.xyz[0]))  # noqa: E501

        # Print new coordinates to file for molA
        line_restart, atom_num_restart = coordinate_swap.write_modified(df_atom_swap, 'B2A', line_start, molB_file, molA_new, atom_num_restart, nameB, nameA, copy.deepcopy(molB.xyz[0]), atom_mapping, atom_order_A, atom_order_B)  # noqa: E501

        if line_restart is not None:
            # Print rest of file after modified residue
            coordinate_swap.write_unmodified(line_restart, molB_file, molA_new, nameB, atom_num_restart, line_start, copy.deepcopy(molB.xyz[0]))  # noqa: E501

        # Print box size
        molA_new.write(molB_file[-1])
        molA_new.close()

        # Rename temp files
        os.rename('A_hybrid_swap.gro', molB_dir + '/confout.gro')
        os.rename('B_hybrid_swap.gro', molA_dir + '/confout.gro')

    def process_top(self):
        """
        Processes the input topologies in order to determine the atoms for alignment in the default GRO swapping
        function. Output as csv files to prevent needing to re-run this step.
        """
        if not os.path.exists('atom_name_mapping.csv'):
            coordinate_swap.create_atom_map(self.gro, self.resname_list, self.swap_rep_pattern)
            atom_name_mapping = pd.read_csv('atom_name_mapping.csv')
        else:
            atom_name_mapping = pd.read_csv('atom_name_mapping.csv')

        if not os.path.exists('residue_connect.csv'):
            df_top = pd.DataFrame()
            for f, file_name in enumerate(self.top):
                # Read file
                input_file = gmx_parser.read_top(file_name, self.resname_list[f])

                # Determine the atom names corresponding to the atom numbers
                start_line, atom_name, atom_num, state = coordinate_swap.get_names(input_file, self.resname_list[f])

                # Determine the connectivity of all atoms
                connect_1, connect_2, state_1, state_2 = [], [], [], []  # Atom 1 and atom 2 which are connected and which state they are dummy atoms  # noqa: E501
                for l, line in enumerate(input_file[start_line:]):  # noqa: E741
                    line_sep = line.split(' ')
                    if line_sep[0] == ';':
                        continue
                    if line_sep[0] == '\n':
                        break
                    while '' in line_sep:
                        line_sep.remove('')
                    if int(line_sep[0]) in atom_num and int(line_sep[1]) in atom_num:
                        num_1 = np.where(atom_num == int(line_sep[0]))[0][0]
                        num_2 = np.where(atom_num == int(line_sep[1]))[0][0]
                        connect_1.append(atom_name[num_1])
                        connect_2.append(atom_name[num_2])
                        state_1.append(state[num_1])
                        state_2.append(state[num_2])

                df = pd.DataFrame({'Resname': self.resname_list[f], 'Connect 1': connect_1, 'Connect 2': connect_2, 'State 1': state_1, 'State 2': state_2})  # noqa: E501
                df_top = pd.concat([df_top, df])
            df_top.to_csv('residue_connect.csv')
        else:
            df_top = pd.read_csv('residue_connect.csv')

        if not os.path.exists('residue_swap_map.csv'):
            df_map = pd.DataFrame()
            for swap in self.swap_rep_pattern:
                # Determine atoms not present in both molecules
                X, Y = [int(swap[0][0]), int(swap[1][0])]
                lam = {X: int(swap[0][1]), Y: int(swap[1][1])}
                swap_name_match = atom_name_mapping[(atom_name_mapping['resname A'].isin([self.resname_list[X], self.resname_list[Y]])) & (atom_name_mapping['resname B'].isin([self.resname_list[X], self.resname_list[Y]]))]  # noqa: E501
                for A, B in zip([X, Y], [Y, X]):
                    # Swapping coordinates from B to A
                    input_A = gmx_parser.read_top(self.top[A], self.resname_list[A])
                    start_line, A_name, A_num, state = coordinate_swap.get_names(input_A, self.resname_list[A])
                    input_B = gmx_parser.read_top(self.top[B], self.resname_list[B])
                    start_line, B_name, B_num, state = coordinate_swap.get_names(input_B, self.resname_list[B])

                    # Determine shared atom names
                    if len(swap_name_match[swap_name_match['resname A'] == self.resname_list[A]]) != 0:
                        common_atoms_A = list(swap_name_match['atom name A'].values)
                    else:
                        common_atoms_A = list(swap_name_match['atom name B'].values)
                    A_only = [x for x in A_name if x not in common_atoms_A]

                    # Seperate real to dummy switches
                    df = coordinate_swap.determine_connection(A_only, swap_name_match, self.resname_list[A], self.resname_list[B], df_top, lam[A])  # noqa: E501

                    df_map = pd.concat([df_map, df])

            df_map.to_csv('residue_swap_map.csv')
