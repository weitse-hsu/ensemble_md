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
Unit tests for the module coordinate_swap.py.
"""
from ensemble_md.utils import coordinate_swap
import numpy as np
import pandas as pd
import os
import mdtraj as md

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


def test_get_dimensions():
    gro = os.path.join(input_path, 'sys.gro')
    f = open(gro, 'r')
    lines = f.readlines()
    f.close()
    vec = coordinate_swap.get_dimensions(lines)
    assert vec == [3.32017, 3.32017, 2.34772, 0.00000, 0.00000, 0.00000, 0.00000, 1.66009, 1.66009]

    # Write a flat file with cubic box dimensions
    f = open('test.gro', 'w')
    f.write('test\n')
    f.write('    1.00000    2.00000    3.00000\n')
    f.close()

    f = open('test.gro', 'r')
    lines = f.readlines()
    f.close()
    vec = coordinate_swap.get_dimensions(lines)
    assert vec == [1.0, 2.0, 3.0]

    os.remove('test.gro')


def test_extract_missing():  # Function changed to extract_missing -- Need to rewrite
    swap_map = pd.read_csv(f'{input_path}/coord_swap/residue_swap_map.csv')
    test_df = coordinate_swap.extract_missing('D2E', 'E2F', swap_map)
    df = pd.read_csv(f'{input_path}/coord_swap/extract_missing.csv')

    for index, row in df.iterrows():
        assert row['Name'] in test_df['Name'].to_list()
        test_row = test_df[test_df['Name'] == row['Name']]
        assert row['Swap'] == test_row['Swap'].values[0]


def test_fix_break():
    broken_mol = md.load(f'{input_path}/coord_swap/broken_mol_1D.gro')
    df_connect = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')
    df_connect_res = df_connect[df_connect['Resname'] == 'C2D']
    test_fix = coordinate_swap.fix_break(broken_mol, 'C2D', [2.74964, 2.74964, 2.74964], df_connect_res)

    broken_mol_3D = md.load(f'{input_path}/coord_swap/broken_mol_3D.gro')
    test_fix_3D = coordinate_swap.fix_break(broken_mol_3D, 'C2D', [2.74964, 2.74964, 2.74964], df_connect_res)

    already_fixed = md.load(f'{input_path}/coord_swap/fixed_mol.gro')
    still_fixed = coordinate_swap.fix_break(already_fixed, 'C2D', [2.74964, 2.74964, 2.74964], df_connect_res)

    fixed_mol = md.load(f'{input_path}/coord_swap/fixed_mol.gro')

    assert (test_fix.xyz == fixed_mol.xyz).all
    assert (test_fix_3D.xyz == fixed_mol.xyz).all
    assert (still_fixed.xyz == fixed_mol.xyz).all


def test_perform_shift():
    broken_mol = md.load(f'{input_path}/coord_swap/broken_mol_1D.gro')

    partial_fix, was_it_fixed, prev_shifted_atoms = coordinate_swap._perform_shift(broken_mol, [2.74964, 2.74964, 2.74964], [[0, 4]], [], 1)  # noqa: E501

    broken_pairs = coordinate_swap._check_break(partial_fix, [[0, 4]])

    assert prev_shifted_atoms == [4]
    assert was_it_fixed is True
    assert len(broken_pairs) == 0

    broken_mol_2D = md.load(f'{input_path}/coord_swap/broken_mol_2D.gro')

    partial_fix, was_it_fixed, prev_shifted_atoms = coordinate_swap._perform_shift(broken_mol_2D, [2.74964, 2.74964, 2.74964], [[0, 4]], [], 2)  # noqa: E501

    broken_pairs = coordinate_swap._check_break(partial_fix, [[0, 4]])

    assert prev_shifted_atoms == [4]
    assert was_it_fixed is True
    assert len(broken_pairs) == 0

    broken_mol_3D = md.load(f'{input_path}/coord_swap/broken_mol_3D.gro')

    partial_fix, was_it_fixed, prev_shifted_atoms = coordinate_swap._perform_shift(broken_mol_3D, [2.74964, 2.74964, 2.74964], [[0, 4]], [], 3)  # noqa: E501

    broken_pairs = coordinate_swap._check_break(partial_fix, [[0, 4]])

    assert prev_shifted_atoms == [4]
    assert was_it_fixed is True
    assert len(broken_pairs) == 0


def test_check_break():
    broken_mol = md.load(f'{input_path}/coord_swap/broken_mol_1D.gro')
    df_connect = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')

    atom_connect_all = df_connect[df_connect['Resname'] == 'C2D']
    mol_top = broken_mol.topology
    resname = 'C2D'
    atom_connect = []
    for i, row in atom_connect_all.iterrows():
        atom_connect.append([row['Connect 1'], row['Connect 2']])

    atom_pairs = []
    for atoms in atom_connect:
        atom_pairs.append(list(mol_top.select(f"resname {resname} and (name {atoms[0]} or name {atoms[1]})")))

    broken_pairs = coordinate_swap._check_break(broken_mol, atom_pairs)

    assert broken_pairs == [[0, 4], [1, 2]] or broken_pairs == [[1, 2], [0, 4]]


def test_get_miss_coord():
    molA_file = f'{input_path}/coord_swap/sim_A/confout_backup.gro'
    molB_file = f'{input_path}/coord_swap/sim_B/confout_backup.gro'
    nameA = 'D2E'
    nameB = 'E2F'

    connection_map = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')
    swap_map = pd.read_csv(f'{input_path}/coord_swap/residue_swap_map.csv')

    molA = md.load(f'{input_path}/coord_swap/sim_A/traj.trr', top=molA_file).slice(-1)
    molB = md.load(f'{input_path}/coord_swap/sim_B/traj.trr', top=molB_file).slice(-1)

    A_dimensions = coordinate_swap.get_dimensions(open(molA_file, 'r').readlines())
    B_dimensions = coordinate_swap.get_dimensions(open(molB_file, 'r').readlines())
    molA = coordinate_swap.fix_break(molA, nameA, A_dimensions, connection_map[connection_map['Resname'] == nameA])
    molB = coordinate_swap.fix_break(molB, nameB, B_dimensions, connection_map[connection_map['Resname'] == nameB])

    df_no_coords = pd.read_csv(f'{input_path}/coord_swap/extract_missing.csv')
    df = pd.read_csv(f'{input_path}/coord_swap/df_atom_swap.csv')
    df_no_nan = df.dropna()

    df_test = coordinate_swap.get_miss_coord(molB, molA, nameB, nameA, df_no_coords, 'A2B',
                                             swap_map[(swap_map['Swap A'] == nameB) & (swap_map['Swap B'] == nameA)])
    df_test = coordinate_swap.get_miss_coord(molA, molB, nameA, nameB, df_test, 'B2A',
                                             swap_map[(swap_map['Swap A'] == nameA) & (swap_map['Swap B'] == nameB)])

    for index, row in df_no_nan.iterrows():
        test_row = df_test[df_test['Name'] == row['Name']]
        assert np.isclose(row['X Coordinates'], test_row['X Coordinates'], atol=10**(-6))
        assert np.isclose(row['Y Coordinates'], test_row['Y Coordinates'], atol=10**(-6))
        assert np.isclose(row['Z Coordinates'], test_row['Z Coordinates'], atol=10**(-6))


def test_process_line():
    file = open(f'{input_path}/coord_swap/sample_process.gro', 'r').readlines()

    line, prev_line = coordinate_swap._process_line(file, 5)

    assert prev_line == ['19GLU', 'H2', '3', '1.800', '4.155', '4.152', '2.0821', '0.4011', '-1.2490\n']
    assert line == ['19GLU', 'H3', '4', '1.833', '4.306', '4.113', '-0.3604', '0.3093', '-1.3761\n']

    line, prev_line = coordinate_swap._process_line(file, 15429)
    assert prev_line == ['4487SOL', 'HW1', '15427', '5.528', '0.500', '4.439', '1.2407', '1.5381', '-0.1116\n']
    assert line == ['4487SOL', 'HW2', '15428', '5.477', '0.613', '4.526', '-0.4379', '1.1843', '-0.6399\n']


def test_print_preamble():
    file = open(f'{input_path}/coord_swap/sim_A/confout_backup.gro', 'r').readlines()
    temp_file = open('test_print_preamble.gro', 'w')

    coordinate_swap.print_preamble(file, temp_file, 5, 1)
    temp_file.close()

    read_temp_file = open('test_print_preamble.gro', 'r').readlines()
    num_atoms_temp_file = int(read_temp_file[1])
    assert num_atoms_temp_file == 2062
    os.remove('test_print_preamble.gro')


def test_write_line():
    test_file = open('test_write_line.gro', 'w')

    new_coord = [3.9165084, 6.3927655, 5.4633074]
    vel = ['0.000', '0.000', '0.000\n']
    atom_num = 11574
    atom_name = 'C10'

    coordinate_swap.write_line(test_file, atom_name, atom_num, vel, new_coord, 36, 'E2F')

    new_coord = [5.4400544, 0.6561325, 8.3108530]
    atom_num = 12264
    atom_name = 'OW'
    coordinate_swap.write_line(test_file, atom_name, atom_num, vel, new_coord, 812, 'SOL')
    test_file.close()

    reopen_test = open('test_write_line.gro', 'r').readlines()

    assert reopen_test[0] == '   36E2F    C1011574   3.9165084   6.3927655   5.4633074   0.000   0.000   0.000\n'
    assert reopen_test[1] == '  812SOL     OW12264   5.4400544   0.6561325   8.3108530   0.000   0.000   0.000\n'
    os.remove('test_write_line.gro')


def test_identify_res():
    swap_map = pd.read_csv(f'{input_path}/coord_swap/residue_swap_map.csv')

    for file_name, real_name in zip(['A-B', 'B-C', 'C-D', 'D-E', 'E-F'], ['A2B', 'B2C', 'C2D', 'D2E', 'E2F']):
        mol = md.load(f'{input_path}/coord_swap/{file_name}.gro')
        residue_options = swap_map['Swap A'].to_list() + swap_map['Swap B'].to_list()
        name = coordinate_swap.identify_res(mol.topology, residue_options)
        assert name == real_name


def test_sep_merge():
    sample_line = ['36B2C', 'N11549', '3.964', '6.464', '6.901', '-0.0888', '-0.6098', '0.8167']
    test_split = coordinate_swap._sep_merge(sample_line)
    assert test_split[0] == '36B2C'
    assert test_split[1] == 'N'
    assert test_split[2] == '11549'

    sample_line = ['36B2C', 'O211557', '3.983', '6.536', '6.608', '-0.2254', '-0.0182', '-0.1860']
    test_split = coordinate_swap._sep_merge(sample_line)
    assert test_split[0] == '36B2C'
    assert test_split[1] == 'O2'
    assert test_split[2] == '11557'


def test_rotate_point_around_axis():
    initial_point = np.array([0.16, 0.19, -0.05])
    vertex = np.array([0, 0, 0])
    axis = np.array([0.15, 0.82, 0.14])
    angle = 0.13
    rotated_point = [0.1693233, 0.18548463, -0.0335421]
    assert (coordinate_swap._rotate_point_around_axis(initial_point, vertex, axis, angle) == rotated_point).all


def test_find_rotation_angle():
    initial_point = np.array([0.16, 0.19, -0.05])
    vertex = np.array([0, 0, 0])
    axis = np.array([0.15, 0.82, 0.14])
    rotated_point = [0.1693233, 0.18548463, -0.0335421]
    angle = 0.13
    test_angle = coordinate_swap._find_rotation_angle(initial_point, vertex, rotated_point, axis)
    assert np.isclose(angle, test_angle, 10**(-5))

    initial_point = np.array([0, 1, 0])
    rotated_point = np.array([0, 1, 0])
    angle = 2*np.pi
    test_angle = coordinate_swap._find_rotation_angle(initial_point, vertex, rotated_point, axis)
    assert np.isclose(angle, test_angle, 10**(-5))


def test_swap_name():  # Update needed
    atom_name_mapping = pd.read_csv(f'{input_path}/coord_swap/atom_name_mapping.csv')
    swap_name_match = atom_name_mapping[(atom_name_mapping['resname A'] == 'A2B') & (atom_name_mapping['resname B'] == 'B2C')]  # noqa: E501

    name_list = ['C2', 'C5', 'DC7', 'HV5']
    flip_name_list = ['C2', 'C5', 'C7', 'H5']
    test_name_list = coordinate_swap._swap_name(name_list, 'B2C', swap_name_match)
    test_flip_name_list = coordinate_swap._swap_name(flip_name_list, 'A2B', swap_name_match)

    assert test_name_list == flip_name_list
    assert test_flip_name_list == name_list


def test_get_names():
    top_files = ['A-B.itp', 'B-C.itp', 'C-D.itp', 'D-E.itp', 'E-F.itp']
    resnames = ['A2B', 'B2C', 'C2D', 'D2E', 'E2F']

    start_lines = [26, 29, 33, 32, 36]
    names = [['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'H1', 'H2', 'H3', 'H4', 'H17', 'DC7', 'HV5', 'HV6', 'HV7'], ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'DC8', 'HV8', 'HV9', 'HV10'], ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C8', 'H1', 'H2', 'H3', 'H4', 'H6', 'H7', 'H8', 'H9', 'H10', 'DC9', 'HV5', 'HV11', 'HV12', 'HV13'], ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C9', 'H1', 'H2', 'H3', 'H5', 'H6', 'H7', 'H11', 'H12', 'H13', 'DC8', 'HV8', 'HV9', 'HV10'], ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'H1', 'H2', 'H3', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'DC10', 'HV4', 'HV14', 'HV15', 'HV16']]  # noqa: E501

    lambda_states = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0]]
    for i, top_file in enumerate(top_files):
        top = open(f'{input_path}/coord_swap/{top_file}', 'r').readlines()
        test_start_line, test_names, test_nums, test_lambda_states = coordinate_swap.get_names(top, resnames[i])
        assert test_start_line == start_lines[i]
        assert test_names == names[i]
        assert test_lambda_states == lambda_states[i]


def test_determine_connection():
    cmpr_df = pd.read_csv(f'{input_path}/coord_swap/residue_swap_map.csv')
    df_top = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')
    atom_name_mapping = pd.read_csv(f'{input_path}/coord_swap/atom_name_mapping.csv')

    A2B_names = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'H1', 'H2', 'H3', 'H4', 'H17', 'DC7', 'HV5', 'HV6', 'HV7']
    B2C_names = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'DC8', 'HV8', 'HV9', 'HV10']  # noqa: E501
    D2E_names = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C9', 'H1', 'H2', 'H3', 'H5', 'H6', 'H7', 'H11', 'H12', 'H13', 'DC8', 'HV8', 'HV9', 'HV10']  # noqa: E501
    E2F_names = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'H1', 'H2', 'H3', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'DC10', 'HV4', 'HV14', 'HV15', 'HV16']  # noqa: E501

    # Determine shared atom names
    swap_name_match = atom_name_mapping[(atom_name_mapping['resname A'] == 'A2B') & (atom_name_mapping['resname B'] == 'B2C')]  # noqa: E501
    if len(swap_name_match[swap_name_match['resname A'] == 'A2B']) != 0:
        common_atoms_A = list(swap_name_match['atom name A'].values)
        common_atoms_B = list(swap_name_match['atom name B'].values)
    else:
        common_atoms_A = list(swap_name_match['atom name B'].values)
        common_atoms_B = list(swap_name_match['atom name A'].values)
    A_only = [x for x in A2B_names if x not in common_atoms_A]
    test_df = coordinate_swap.determine_connection(A_only, swap_name_match, 'A2B', 'B2C', df_top, 1)
    select_cmpr_df = cmpr_df[(cmpr_df['Swap A'] == 'A2B') & (cmpr_df['Swap B'] == 'B2C')]
    for col in ['Anchor Atom Name A', 'Anchor Atom Name B', 'Alignment Atom A', 'Alignment Atom B', 'Angle Atom A', 'Angle Atom B', 'Missing Atom Name']:  # noqa: E501
        assert test_df[col].to_list()[0] == select_cmpr_df[col].to_list()[0]

    B_only = [x for x in B2C_names if x not in common_atoms_B]
    test_df = coordinate_swap.determine_connection(B_only, swap_name_match, 'B2C', 'A2B', df_top, 0)
    select_cmpr_df = cmpr_df[(cmpr_df['Swap B'] == 'A2B') & (cmpr_df['Swap A'] == 'B2C')]
    for col in ['Anchor Atom Name A', 'Anchor Atom Name B', 'Alignment Atom A', 'Alignment Atom B', 'Angle Atom A', 'Angle Atom B', 'Missing Atom Name']:  # noqa: E501
        assert test_df[col].to_list()[0] == select_cmpr_df[col].to_list()[0]

    swap_name_match = atom_name_mapping[(atom_name_mapping['resname A'] == 'D2E') & (atom_name_mapping['resname B'] == 'E2F')]  # noqa: E501
    if len(swap_name_match[swap_name_match['resname A'] == 'D2E']) != 0:
        common_atoms_A = list(swap_name_match['atom name A'].values)
        common_atoms_B = list(swap_name_match['atom name B'].values)
    else:
        common_atoms_A = list(swap_name_match['atom name B'].values)
        common_atoms_B = list(swap_name_match['atom name A'].values)
    A_only = [x for x in D2E_names if x not in common_atoms_A]
    B_only = [x for x in E2F_names if x not in common_atoms_B]
    test_df = coordinate_swap.determine_connection(A_only, swap_name_match, 'D2E', 'E2F', df_top, 1)
    select_cmpr_df = cmpr_df[(cmpr_df['Swap A'] == 'D2E') & (cmpr_df['Swap B'] == 'E2F')]
    for col in ['Anchor Atom Name A', 'Anchor Atom Name B', 'Alignment Atom A', 'Alignment Atom B', 'Angle Atom A', 'Angle Atom B', 'Missing Atom Name']:  # noqa: E501
        assert test_df[col].to_list()[0] == select_cmpr_df[col].to_list()[0]

    test_df = coordinate_swap.determine_connection(B_only, swap_name_match, 'E2F', 'D2E', df_top, 0)
    select_cmpr_df = cmpr_df[(cmpr_df['Swap A'] == 'E2F') & (cmpr_df['Swap B'] == 'D2E')]
    for col in ['Anchor Atom Name A', 'Anchor Atom Name B', 'Alignment Atom A', 'Alignment Atom B', 'Angle Atom A', 'Angle Atom B', 'Missing Atom Name']:  # noqa: E501
        assert test_df[col].to_list()[0] == select_cmpr_df[col].to_list()[0]


def test_create_atom_map():
    gro = [f'{input_path}/coord_swap/A-B.gro', f'{input_path}/coord_swap/B-C.gro', f'{input_path}/coord_swap/C-D.gro', f'{input_path}/coord_swap/D-E.gro', f'{input_path}/coord_swap/E-F.gro']  # noqa: E501
    names = ['A2B', 'B2C', 'C2D', 'D2E', 'E2F']
    swap_pattern = [[[0, 1], [1, 0]], [[1, 1], [2, 0]], [[2, 1], [3, 0]], [[3, 1], [4, 0]]]

    atom_name_mapping_true = pd.read_csv(f'{input_path}/coord_swap/atom_name_mapping.csv')
    coordinate_swap.create_atom_map(gro, names, swap_pattern)
    atom_name_mapping_test = pd.read_csv('atom_name_mapping.csv')
    assert (atom_name_mapping_true == atom_name_mapping_test).all
    os.remove('atom_name_mapping.csv')
