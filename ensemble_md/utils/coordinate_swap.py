####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by                                                    #
#      - Anika Friedman <anika.friedman@colorado.edu>              #
#      - Wei-Tse Hsu <wehs7661@colorado.edu>                       #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################
"""
The :obj:`.coordinate_swap` module provides functions for swapping coordinates
in a MT-REXEE simulation.
"""
import re
import copy
import mdtraj as md
import numpy as np
import pandas as pd
from itertools import product


def get_dimensions(file):
    """
    Determines the dimensions of the cubic box based on the input GRO file.

    Parameters
    ----------
    file : list
        A list of strings containing all of the lines in the input GRO file.

    Returns
    -------
    dim_vector : list
        A list of floats containing the dimensions of the cubic box.
    """
    box_raw = file[-1].split(' ')
    while '' in box_raw:
        box_raw.remove('')

    dim_vector = [float(i) for i in box_raw]
    return dim_vector


def extract_missing(nameA, nameB, swap_map):
    """
    Determine the atoms which are common, which are switched between dummy and real atoms,
    and which are unique between the two input molecules.

    Parameters  <---- CHANGE ME
    ----------
    molA_file : list
        A list of strings containing lines of the GRO file for molecule A.
    molB_file : list
        A list of strings containing lines of the GRO file for molecule B.
    nameA : str
        The name of the residue being altered in molecule A.
    nameB : str
        The name of the residue being altered in molecule B.

    Returns
    -------
    df : pandas.DataFrame
        A pandas DataFrame containing the atoms which are not the same between the two molecules
        and how they are changed.
    """
    # Gather missing atoms for each swap
    A2B = swap_map[(swap_map['Swap A'] == nameB) & (swap_map['Swap B'] == nameA)]
    missing_A2B = []
    for r, row in A2B.iterrows():
        atoms = row['Missing Atom Name']
        split_atoms = atoms.split(' ')
        for a in split_atoms:
            missing_A2B.append(a)

    B2A = swap_map[(swap_map['Swap A'] == nameA) & (swap_map['Swap B'] == nameB)]
    missing_B2A = []
    for r, row in B2A.iterrows():
        atoms = row['Missing Atom Name']
        split_atoms = atoms.split(' ')
        for a in split_atoms:
            missing_B2A.append(a)

    # Determine the swaps for each transformation
    df = pd.concat([pd.DataFrame({'Name': missing_A2B, 'Swap': 'A2B'}), pd.DataFrame({'Name': missing_B2A, 'Swap': 'B2A'})])  # noqa: E501
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    return df


def fix_break(mol, resname, box_dimensions, atom_connect_all):
    """
    Determines if there are any breaks across periodic boundary conditions in the residue of interest and
    shifts the molecule across those boudaries to make it whole.

    Parameters
    ----------
    mol : :func:`mdtraj.Trajectory` object
        Trajectory for the molecule or interest.
    resname : str
        The name of the residue of interest for coordinate swapping.
    box_dimensions : list
        Dimensions of the periodic box for the system of interest.
    atom_connect_all : pandas.DataFrame
        A pandas DataFrame which contains the name of all atoms which are connected to one another
        in the residue of interest

    Returns
    -------
    mol : :func:`mdtraj.Trajectory` object
        A :func:`mdtraj.Trajectory` object with all breaks in the necessary portions of the molecule fixed.
    """
    mol_top = mol.topology

    atom_connect = []
    for i, row in atom_connect_all.iterrows():
        atom_connect.append([row['Connect 1'], row['Connect 2']])

    atom_pairs = []
    for atoms in atom_connect:
        atom_pairs.append(list(mol_top.select(f"resname {resname} and (name {atoms[0]} or name {atoms[1]})")))

    # Find any broken bonds
    broken_pairs = _check_break(mol, atom_pairs)

    if len(broken_pairs) == 0:
        print('No breaks found')
        return mol
    else:
        print('Fixing break')

    iter = 0  # Keep track of while loop iterations
    shift_atom = []
    while len(broken_pairs) > 0:
        # Prevent infinite loop
        if iter > len(atom_pairs):
            raise Exception('Break could not be fixed')
        iter += 1

        # Fix this break
        mol, fixed, shift_atom = _perform_shift(mol, box_dimensions, broken_pairs, shift_atom, 1)
        if fixed:
            broken_pairs = _check_break(mol, atom_pairs)
            continue
        else:
            mol, fixed, shift_atom = _perform_shift(mol, box_dimensions, broken_pairs, shift_atom, 2)
            if fixed:
                broken_pairs = _check_break(mol, atom_pairs)
                continue
            else:
                mol, fixed, shift_atom = _perform_shift(mol, box_dimensions, broken_pairs, shift_atom, 3)
                if fixed:
                    broken_pairs = _check_break(mol, atom_pairs)
                    continue
                else:
                    raise Exception('Break could not be fixed')

    return mol


def _perform_shift(mol, box_dimensions, broken_pairs_init, prev_shift_atom, num_shift_dimensions):
    """
    Shifts the input trajectory across the periodic boundaries in 1D.

    Parameters
    ----------
    mol : :func:`mdtraj.Trajectory` object
        A :func:`mdtraj.Trajectory` object with the original coordinates prior to the shift.
    box_dimensions : list
        Dimensions of the periodic boundary box.
    broken_pairs_init : int
        Which pairs of atoms were found to be broken.
    prev_shift_atom : int
        Which atoms have already been shifted so we don't undo what we've done.
    num_shift_dimensions : int
        Whether the shift should be attempted in 1, 2, or 3 dimensions

    Returns
    -------
    mol : :func:`mdtraj.Trajectory` object
        Trajectory with the new coordinates.
    fixed : bool
        A boolean indicating whether the break was actually fixed.
    prev_shift_atom : int
        Which atoms have already been shifted so we don't undo what we've done
    """
    atom_pair = broken_pairs_init[0]
    broken_atom = atom_pair[1]
    if broken_atom in prev_shift_atom:
        broken_atom = atom_pair[0]
    fixed = False
    if num_shift_dimensions == 1:
        shift_combos = np.concatenate((np.identity(3), -1*np.identity(3)), axis=0)
    elif num_shift_dimensions == 2:
        shift_combos = [[1, 1, 0],
                        [1, -1, 0],
                        [-1, 1, 0],
                        [-1, -1, 0],
                        [0, 1, 1],
                        [0, 1, -1],
                        [0, -1, 1],
                        [0, -1, -1],
                        [1, 0, 1],
                        [1, 0, -1],
                        [-1, 0, 1],
                        [-1, 0, -1]]
    else:
        shift_combos = product([1, -1], [1, -1], [1, -1])
    for shift_dir in shift_combos:  # Try all combos of shift direction
        mol.xyz[0, broken_atom, 0] = mol.xyz[0, broken_atom, 0] + (shift_dir[0] * box_dimensions[0])
        mol.xyz[0, broken_atom, 1] = mol.xyz[0, broken_atom, 1] + (shift_dir[1] * box_dimensions[1])
        mol.xyz[0, broken_atom, 2] = mol.xyz[0, broken_atom, 2] + (shift_dir[2] * box_dimensions[2])
        dist_check = md.compute_distances(mol, atom_pairs=[atom_pair], periodic=False)
        if dist_check > 0.2:  # Didn't work so reverse and try again
            mol.xyz[0, broken_atom, 0] = mol.xyz[0, broken_atom, 0] - (shift_dir[0] * box_dimensions[0])
            mol.xyz[0, broken_atom, 1] = mol.xyz[0, broken_atom, 1] - (shift_dir[1] * box_dimensions[1])
            mol.xyz[0, broken_atom, 2] = mol.xyz[0, broken_atom, 2] - (shift_dir[2] * box_dimensions[2])
        else:  # Yay fixed break
            fixed = True
            break
    if fixed:
        prev_shift_atom.append(broken_atom)
    return mol, fixed, prev_shift_atom


def _check_break(mol, atom_pairs):
    """
    Determines whether a break is present between the atom pairs of interest.

    Parameters
    ----------
    mol : :func:`mdtraj.Trajectory` object
        Trajectory to examine for breaks.
    atom_pairs : int
        The atom number pairs which should be connected in the residue of interest.

    Returns
    -------
    broken_pairs : int
        Which pairs of atoms have breaks across the periodic boundary.
    """
    dist = md.compute_distances(mol, atom_pairs=atom_pairs, periodic=False)
    broken_pairs = []
    for i, d in enumerate(dist[0]):
        if d > 0.2:
            broken_pairs.append(atom_pairs[i])
    return broken_pairs


def get_miss_coord(mol_align, mol_ref, name_align, name_ref, df_atom_swap, dir, df_swap):
    """
    Gets coordinates for the missing atoms after the conformational swap

    Parameters
    ----------
    mol_align : :func:`mdtraj.Trajectory` object
        Trajectory for the molecule being aligned.
    mol_ref : :func:`mdtraj.Trajectory` object
        Trajectory for the reference molecule.
    name_align : str
        The resname for the molecule which is being aligned.
    name_ref : str
        The resname for the molecule which has the reference coordinates.
    df_atom_swap : pandas.DataFrame
        A pandas DataFrame that contains the following information for each at omthat is no in the common list:

          - The name of the atom
          - The atom number
          - The atom element
          - Whether the atom is switchin between dummy and real or missing
          - The swap direction (same as input)
          - Thether the final atom type is real or dummy
    dir : str
        The swapping direction.
    df_swap : pandas.DataFrame
        Swapping map for the given conformational swap direction to determine which atoms
        to use for anchor, alignment, and angle determination.

    Returns
    -------
    df_atom_swap : pandas.DataFrame
        Same dataframe as the input, but with coordinates for the missing atoms.
    """
    # Create a new column for coordinates if one does not exist
    if 'X Coordinates' not in df_atom_swap.columns:
        df_atom_swap['X Coordinates'] = np.nan
        df_atom_swap['Y Coordinates'] = np.nan
        df_atom_swap['Z Coordinates'] = np.nan

    if len(df_swap.index) == 0:
        return df_atom_swap

    for i, row in df_swap.iterrows():
        conn_align = [row['Anchor Atom Name A'], row['Alignment Atom A']]
        conn_ref = [row['Anchor Atom Name B'], row['Alignment Atom B']]
        miss_names_select = row['Missing Atom Name'].split(' ')
        geom_fix_select = [miss_names_select[0], row['Anchor Atom Name A'], row['Angle Atom A']]
        # Limit to region of interest
        align_select = []
        for a in conn_align + miss_names_select + geom_fix_select:
            atom = mol_align.topology.select(f'resname {name_align} and name {a}')[0]
            if atom not in align_select:
                align_select.append(atom)
        align_select.sort()
        ref_select = mol_ref.topology.select(f"resname {name_ref} and (name {conn_ref[0]} or name {conn_ref[1]} or name {row['Angle Atom B']})")  # noqa: E501

        mol_ref_select = copy.deepcopy(mol_ref.atom_slice(ref_select))
        mol_align_select = copy.deepcopy(mol_align.atom_slice(align_select))

        # Select atoms in connection bond
        conn0_align = mol_align_select.topology.select(f'resname {name_align} and name {conn_align[0]}')[0]
        conn1_align = mol_align_select.topology.select(f'resname {name_align} and name {conn_align[1]}')[0]
        conn0_ref = mol_ref_select.topology.select(f'resname {name_ref} and name {conn_ref[0]}')[0]
        conn1_ref = mol_ref_select.topology.select(f'resname {name_ref} and name {conn_ref[1]}')[0]

        # Step 1: Perform translation to align point 1
        shift_vec = mol_ref_select.xyz[0, conn0_ref, :] - mol_align_select.xyz[0, conn0_align, :]
        for a in range(mol_align_select.n_atoms):
            mol_align_select.xyz[0, a, :] = mol_align_select.xyz[0, a, :] + shift_vec

        # Step 2: Perform rotational motion to line up point 2
        ref_vec = mol_ref_select.xyz[0, conn1_ref, :] - mol_ref_select.xyz[0, conn0_ref, :]  # defing 0-1 vector in ref  # noqa: E501
        align_vec = mol_align_select.xyz[0, conn1_align, :] - mol_ref_select.xyz[0, conn0_ref, :]  # defing 0-1 vector in align  # noqa: E501
        axis_rot = np.cross(ref_vec/np.linalg.norm(ref_vec), (align_vec/np.linalg.norm(align_vec)))  # Perpendicular vector to ref and align vectors  # noqa: E501
        theta = _find_rotation_angle(mol_align_select.xyz[0, conn1_align, :], mol_ref_select.xyz[0, conn0_ref, :], mol_ref_select.xyz[0, conn1_ref, :], axis_rot)  # noqa: E501

        for a in range(mol_align_select.n_atoms):
            if a != conn0_align:
                mol_align_select.xyz[0, a, :] = _rotate_point_around_axis(mol_align_select.xyz[0, a, :], mol_ref_select.xyz[0, conn0_ref, :], axis_rot, theta)  # noqa: E501

        # Step 3: Rotate around 0-1 bond to get the correct angle for connecting atom
        angle_atoms = [
            mol_align.topology.select(f'resname {name_align} and name {geom_fix_select[0]}')[0],
            mol_align.topology.select(f'resname {name_align} and name {geom_fix_select[1]}')[0],
            mol_align.topology.select(f'resname {name_align} and name {geom_fix_select[2]}')[0]
        ]
        internal_angle = _compute_angle([mol_align.xyz[0, angle_atoms[0], :], mol_align.xyz[0, angle_atoms[1], :], mol_align.xyz[0, angle_atoms[2], :]])  # noqa: E501
        change_atoms = mol_align_select.topology.select(f'resname {name_align} and name {geom_fix_select[0]}')
        axis_rot = mol_align_select.xyz[0, conn0_align, :] - mol_align_select.xyz[0, conn1_align, :]  # noqa: E501
        axis_rot = axis_rot / np.linalg.norm(axis_rot)

        # Determine angle of rotation to get the right internal angle
        min_dev_angle = 3
        init_coords = mol_align_select.xyz[0, change_atoms, :][0].copy()
        constant_coords = [
            mol_align_select.xyz[0, mol_align_select.topology.select(f'resname {name_align} and name {geom_fix_select[1]}')[0], :],  # noqa: E501
            mol_ref_select.xyz[0, mol_ref_select.topology.select(f"resname {name_ref} and name {row['Angle Atom B']}")[0], :]  # noqa: E501
        ]
        for theta in np.linspace(0, 2 * np.pi, num=10):
            new_coors = np.zeros((3, 3))
            new_coors[0, :] = _rotate_point_around_axis(init_coords, mol_ref_select.xyz[0, conn0_ref, :], axis_rot, theta)  # noqa: E501
            new_coors[1:, :] = constant_coords
            iangle = _compute_angle(new_coors)
            if abs(iangle - internal_angle) < min_dev_angle:
                theta_min = theta
                min_dev_angle = abs(iangle - internal_angle)
                if min_dev_angle < 0.05:
                    break
        # Perform the rotation
        for a in range(mol_align_select.n_atoms):
            if a != conn0_align:
                mol_align_select.xyz[0, a, :] = _rotate_point_around_axis(mol_align_select.xyz[0, a, :], mol_ref_select.xyz[0, conn0_ref, :], axis_rot, theta_min)  # noqa: E501

        # Add coordinates to df
        for r in range(len(df_atom_swap.index)):
            if df_atom_swap.iloc[r]['Swap'] == dir:
                for name in miss_names_select:
                    if df_atom_swap.iloc[r]['Name'] == name:
                        a = mol_align_select.topology.select(f'name {name}')
                        df_atom_swap.at[r, 'X Coordinates'] = mol_align_select.xyz[0, a, 0]
                        df_atom_swap.at[r, 'Y Coordinates'] = mol_align_select.xyz[0, a, 1]
                        df_atom_swap.at[r, 'Z Coordinates'] = mol_align_select.xyz[0, a, 2]
                        continue

    return df_atom_swap


def _compute_angle(coords):
    """
    Computes the angle between two vectors.

    Parameters
    ----------
    coords : list
        A list of numpy arrays containing the XYZ coordinates of 3 points, for which the angle 1-2-3 is to be computed. # noqa: E501

    Returns
    -------
    angle : int
        Angle in radians between the two points.
    """
    vec1 = coords[0] - coords[1]
    vec2 = coords[2] - coords[1]

    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    return angle


def _process_line(mol_file, i):
    """
    Seperates the line by spaces and corrects for when the atom number merges the atom name

    Parameters
    ----------
    mol_file : list
        A list of strings containing the lines of the input GRO file.
    i : int
        The index the line in the file.

    Returns
    -------
    line : str
        Line i in the file
    prev_line : str
        Line i-1 in the file
    """
    # Load previous line to determine identity of previous atom
    prev_line = mol_file[i-1].split(' ')
    while ("" in prev_line):
        prev_line.remove("")

    # Load current line in file and seperate
    line = mol_file[i].split(' ')
    while ("" in line):
        line.remove("")

    # Control for atom name and atom number merging
    if len(line[1]) > 5:
        line = _sep_merge(line)
    if i > 2 and len(prev_line[1]) > 5:
        prev_line = _sep_merge(prev_line)

    return line, prev_line


def print_preamble(old_file, new_file, num_add_names, num_rm_names):
    """
    Prints the preamble of the gro file before the atom specifications begin.

    Parameters
    ----------
    old_file : list
        A list of strings containing lines from the original GRO file.
    new_file : file-like object
        A writable file-like object where the modified content is written.
    num_add_names : int
        The number of atoms to be added to the pre-swap molecule.
    num_rm_names : int
        The number of atoms to be added to the post-swap molecule.

    Returns
    -------
    line_start : int
        The line number to start when you continue to read the file.
    """
    for i in range(len(old_file)):
        if not re.match(r'[ \t]', old_file[i]) and not re.match('^[0-9]+$', old_file[i]):
            new_file.write(old_file[i])
        else:
            num_atom = int(float(old_file[i]) + num_add_names - num_rm_names)
            new_file.write(f'{num_atom}\n')
            line_start = i + 1
            break
    return line_start


def write_line(mol_new, atom_name, atom_num, vel, coor, resnum, nameB):
    """
    Writes a line in the file in which some parameter needs to be changed.

    Parameters
    ----------
    mol_new : file-like object
        A writable file-like object where the formatted molecular data is written.
    atom_name : str
        Name of atom to write
    atom_num : int
        The new atom number to be assigned to the atom in this line.
    vel : list
        The list of velocities in the x, y, and z directions to be assigned to the atom in this line.
    coor : list
        The list coordinates in the x, y, and z directions for the atom in this line.
    resnum : int
        The new residue number if it changes. The default is :code:`None`:.
    nameB : str
        The new residue name if if changes from what was in the previous file. The default is :code:`None`:.
    """
    coor = ["{:.7f}".format(coor[0]), "{:.7f}".format(coor[1]), "{:.7f}".format(coor[2])]
    mol_new.write(
        f'{resnum}{nameB}'.rjust(8, ' ') +
        atom_name.rjust(7, ' ') +
        str(atom_num).rjust(5, ' ') +
        str(coor[0]).rjust(12, ' ') +
        str(coor[1]).rjust(12, ' ') +
        str(coor[2]).rjust(12, ' ') +
        vel[0].rjust(8, ' ') +
        vel[1].rjust(8, ' ') +
        vel[2].rjust(9, ' ')
    )


def identify_res(mol_top, resname_options):
    """
    Determines which of the potential residues of interest are in this molecule.

    Parameters
    ----------
    mol_top : mdtraj topology
        The molecule topology.
    resname_options : list
        The potential residue names which may be in the moleucle.

    Returns
    -------
    resname : str
        The name of the residue of interest which is in the molecule.
    """
    for name in resname_options:
        if len(mol_top.select(f"resname {name}")) != 0:
            resname = name
            break

    return resname


def _sep_merge(line):
    """
    Seperates two GRO file columns which no longer have a space sperating them.

    Parameters
    ----------
    line : list of str
        The line contents seperated by spaces.

    Returns
    -------
    temp_line : list
        The line contents with the merged columns.
    """
    temp_line = [line[0]]
    merged = [*line[1]]
    atom_name = "".join(merged[:-5])
    atom_num = "".join(merged[-5:])
    temp_line.append(atom_name)
    temp_line.append(atom_num)
    for n in line[2:]:
        temp_line.append(n)
    return temp_line


def _rotate_point_around_axis(point, vertex, axis, angle):
    """
    Rotates a 3D point around an arbitrary axis.

    Parameters
    ----------
    point : numpy.ndarray
        The XYZ coordinates of the point to rotate. The shape should be (1, 3).
    start : numpy.ndarray
        The XYZ coordinates of the vertex of the axis of rotation. The shape should be (1, 3).
    axis : numpy.ndarray
        The axis of rotation. The shape should be (1, 3).
    angle : float
        The angle by which to rotate the point, in radians.

    Returns
    -------
    rotated point : numpy.ndarray
        The XYZ coordinates of the new rotated point. The shape should be (1, 3).
    """
    # Transformation matrix to origin
    T = np.eye(4)
    T[0:3, 3] = -vertex
    # Reverse the transformation to origin
    T_r = np.eye(4)
    T_r[0:3, 3] = vertex
    # Normalize rotational axis
    axis = axis / np.linalg.norm(axis)
    a, b, c = axis
    d = np.sqrt(b ** 2 + c ** 2)

    # Rotation about the x axis
    Rx = [[1, 0, 0, 0], [0, c / d, -b / d, 0], [0, b / d, c / d, 0], [0, 0, 0, 1]]
    Rx_r = [[1, 0, 0, 0], [0, c / d, b / d, 0], [0, -b / d, c / d, 0], [0, 0, 0, 1]]

    # Rotation about the y axis
    Ry = [[d, 0, -a, 0], [0, 1, 0, 0], [a, 0, d, 0], [0, 0, 0, 1]]
    Ry_r = [[d, 0, a, 0], [0, 1, 0, 0], [-a, 0, d, 0], [0, 0, 0, 1]]

    # Rotation about the z axis
    Rz = [[np.cos(angle), np.sin(angle), 0, 0], [-np.sin(angle), np.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    # Initial position
    x0, y0, z0 = point
    init_arry = [x0, y0, z0, 1]

    # Perform the transofmration
    final_arry = T_r@Rx_r@Ry_r@Rz@Ry@Rx@T@init_arry

    # Get the corrdinates for the rotated point
    rotated_point = final_arry[0:3]

    return rotated_point


def _find_rotation_angle(initial_point, vertex, rotated_point, axis):
    """
    Determines the angle of rotation around an arbitrary axis in 3D space.

    Parameters
    ----------
    initial_point: numpy.ndarray
        The initial point coordinates.
    rotated_point: numpy.ndarray
        The rotated point coordinates.
    axis: A numpy array
        The axis of rotation.

    Returns
    -------
    angle: int
        The angle of rotation in radians.
    """
    # Normalize axis and seperate coordinates
    u, v, w = axis / np.linalg.norm(axis)
    x0, y0, z0 = initial_point
    a, b, c = vertex

    # Determine the rotated point with the same bond length as the initial point
    rotated_point_axis = rotated_point - vertex
    norm_rot_point_vec = rotated_point_axis / np.linalg.norm(rotated_point_axis)
    final_bond_length = np.sqrt((x0 - a) ** 2 + (y0 - b) ** 2 + (z0 - c) ** 2)
    rotated_point_standard = vertex + final_bond_length*norm_rot_point_vec
    x, y, z = rotated_point_standard  # Change to point with same bond length along this vector

    Q = np.sqrt(v ** 2 + w ** 2)
    B = ((w / Q) * (y0 - b)) - ((v / Q) * (z0 - c))
    A = Q * (x0 - a) - u * (((v / Q) * (y0 - b)) + ((w / Q) * (z0 - c)))
    C = u * (x0 - a) + Q * (((v / Q) * (y0 - b)) + ((w / Q) * (z0 - c)))
    L = np.arctan2(B, A)

    # Determine the rotational angle to line-up point
    cos_arg = (x - a - u * C) / (Q * np.sqrt(A ** 2 + B ** 2))
    if cos_arg > 1:
        if (cos_arg - 1) < 0.001:
            cos_arg = 1
        else:
            raise Exception(f'Invalid cos argument: {cos_arg}')
    if cos_arg < -1:
        if (cos_arg + 1) < 0.001:
            cos_arg = -1
        else:
            raise Exception(f'Invalid cos argument: {cos_arg}')

    arc_cos = np.arccos(cos_arg)
    angle = L + arc_cos

    # Test point
    new_pt = _rotate_point_around_axis(initial_point, vertex, axis / np.linalg.norm(axis), angle)
    if abs(new_pt[1] - y) < 0.0001:
        return angle
    else:
        arc_cos = 2*np.pi - arc_cos
        angle = L + arc_cos
        if angle > 2*np.pi:
            angle = angle - 2*np.pi

    return angle


def write_modified(df_atom_swap, swap, line_start, orig_file, new_file, atom_num_init, old_res_name, new_res_name, orig_coords, atom_mapping, atom_order, old_atom_order):  # noqa: E501
    """
    Writes a new GRO file.

    Parameters
    ----------
    df_atom_swap : pandas.DataFrame
        Master DataFrame containing info on the atoms which will change before and after the swap.
    swap : str
        The swapping direction.
    r_swap : str
        Reverse of the swapping direction.
    line_start : int
        The line number where we start reading the file.
    orig_file : list
        List of strings containing the content of the pre-swap file to read from.
    new_file : file-like object
        Temporary file to write new coordinates.
    old_res_name : str
        Residue name from before the swap.
    new_res_name : str
        Residue name for after the swap.
    orig_coords : numpy.ndarray
        Coordinates for all atoms in the system before the swap.
    miss : list
        Residues which are needed after the swap which are not present before the swap.
    atom_order : list of str
        List of the atom names in the order they appear in the GRO file
    """
    # Add vero velocity to all atoms
    vel = ['0.000', '0.000', '0.000\n']

    # Determine the direction of the swap
    if len(atom_mapping[(atom_mapping['resname A'] == old_res_name) & (atom_mapping['resname B'] == new_res_name)]) != 0:  # noqa: E501
        atom_map = atom_mapping[(atom_mapping['resname A'] == old_res_name) & (atom_mapping['resname B'] == new_res_name)]  # noqa: E501
        old_res_side = 'A'
        new_res_side = 'B'
    else:
        atom_map = atom_mapping[(atom_mapping['resname B'] == old_res_name) & (atom_mapping['resname A'] == new_res_name)]  # noqa: E501
        old_res_side = 'B'
        new_res_side = 'A'

    # Process input lines
    line, prev_line = _process_line(orig_file, line_start)

    # Seperate resname from resnumber
    res_i = [*line[0]]
    resnum = "".join(res_i[:-3])

    line_restart = None
    atom_num = copy.deepcopy(atom_num_init)
    for atom in atom_order:
        # If the atom is not missing
        if atom in atom_map[f'atom name {new_res_side}'].values:
            # Determine the name of the equivalent atom in the old file
            equivalent_atom_name = atom_map[atom_map[f'atom name {new_res_side}'] == atom][f'atom name {old_res_side}'].values[0]  # noqa: E501

            # Determine index of this atom in the old file coordinates
            coor_index = old_atom_order.index(equivalent_atom_name) + atom_num_init - 1

            write_line(new_file, atom, atom_num, vel, orig_coords[coor_index], resnum, new_res_name)
        # If the atom is missing
        else:
            # Get atom coordinates
            select_row = df_atom_swap[(df_atom_swap['Swap'] == swap) & (df_atom_swap['Name'] == atom)]
            x = float(select_row['X Coordinates'].values[0])
            y = float(select_row['Y Coordinates'].values[0])
            z = float(select_row['Z Coordinates'].values[0])

            write_line(new_file, atom, atom_num, vel, [x, y, z], resnum, new_res_name)
        atom_num += 1

    for i in range(line_start, len(orig_file)-1):
        # Process input lines
        line, prev_line = _process_line(orig_file, i)

        # Seperate resname from resnumber
        res_i = [*line[0]]
        resname = "".join(res_i[-3:])

        # Check if at residue to modify
        if resname != old_res_name:
            line_restart = i
            break

    return line_restart, atom_num


def write_unmodified(line_start, orig_file, new_file, old_res_name, atom_num, preamble_legth, coords):  # noqa: E501
    """
    Writes lines before modification to new file

    Parameters
    ----------
    line_start : int
        The line number where we start reading the file.
    orig_file : list
        List of strings containing the content of the pre-swap file to read from.
    new_file : file-like object
        Temporary file to write new coordinates.
    old_res_name : str
        Residue name from before the swap.
    atom_num : int
        Initial atom number on which to start writing
    """
    vel = ['0.000', '0.000', '0.000\n']

    line_restart = np.nan
    for i in range(line_start, len(orig_file)-1):
        # Process input lines
        line, prev_line = _process_line(orig_file, i)

        # Seperate resname from resnumber
        res_i = [*line[0]]
        resname = "".join(res_i[-3:])
        resnum = "".join(res_i[:-3])
        # Check if at residue to modify
        if resname == old_res_name:
            line_restart = i
            atom_num_restart = atom_num
            break
        else:
            coor_index = i - preamble_legth
            write_line(new_file, line[1], atom_num, vel, coords[coor_index], resnum, resname)
        atom_num += 1
    if line_restart is not np.nan:
        return line_restart, atom_num_restart


def _sep_num_element(atom_name):
    """
    Seperate the atom name into the element and the atom number

    Parameters
    ----------
    atom_name : str
        Name of the atom to be seperated

    Returns
    -------
    element : str
        The letter element identifier without dummy atom modifiers
    num : str or np.nan
        The atom number
    """
    if len(re.findall(r'[0-9]+', atom_name)) == 0:
        num = np.nan
    else:
        num = re.findall(r'[0-9]+', atom_name)[0]
    atom_identifier = re.findall(r'[a-zA-Z]+', atom_name)[0]
    if list(atom_identifier)[0] == 'D':
        element = list(atom_identifier)[1]
        if len(list(atom_identifier)) > 2:
            extra = ''.join(list(atom_identifier)[2:])
        else:
            extra = ''
    else:
        element = list(atom_identifier)[0]
        if len(list(atom_identifier)) > 1:
            extra = ''.join(list(atom_identifier)[1:])
        else:
            extra = ''
    if 'V' in extra:
        extra = extra.strip('V')
    return element, num, extra


def _swap_name(init_atom_names, new_resname, swap_name_match):
    """
    Determines the corresponding atom name in new molecule

    Parameters
    ----------
    init_atom_names : list
        A list of atom names in the original molecule.
    new_resname : str
        Resname for the new molecule.
    swap_name_match : pd.DataFrame
        Names for all common atoms in the target molecule and their name in the current molecule

    Returns
    -------
    new_atom_names : list
        A list of atom names in the new molecule.
    """
    if len(swap_name_match[swap_name_match['resname A'] == new_resname]) != 0:
        new_state = 'A'
        old_state = 'B'
    else:
        old_state = 'A'
        new_state = 'B'
    new_atom_names = []
    for atom in init_atom_names:
        # Find the atom in the map
        if len(swap_name_match[swap_name_match[f'atom name {old_state}'] == atom]) == 0:
            raise Exception(f'Compatible atom could not be found for {atom}')
        new_atom_names.append(swap_name_match[swap_name_match[f'atom name {old_state}'] == atom][f'atom name {new_state}'].values[0])  # noqa: E501
    return new_atom_names


def get_names(input, resname):
    """
    Determines the names of all atoms in the topology and which :math:`lambda` state for which they are dummy atoms.

    Parameters
    ----------
    input : list
        A list of strings containing the lines of the input topology.
    resname : str
        Name of residue of interest for which to extract atom names
    Returns
    -------
    start_line : int
        The next line to start reading from the topology.
    atom_name : list
        All atom names in the topology corresponding to the residue of interest.
    atom_num : list
        The atom numbers corresponding to the atom names in atom_name
    state : list
        The state that the atom is a dummy atom (:math:`lambda=0`, :math:`lambda=1`, or -1 if nevver dummy).
    """
    atom_section = False
    atom_name, atom_num, state = [], [], []
    for l, line in enumerate(input):  # noqa: E741
        if atom_section:
            line_sep = line.split(' ')
            if line_sep[0] == ';':
                continue
            while '' in line_sep:
                line_sep.remove('')
            if line_sep[0] == '\n' or line == '[ bonds ]\n':
                start_line = l+2
                break
            if line_sep[3] == resname:
                atom_name.append(line_sep[4])
                atom_num.append(int(line_sep[0]))
                if float(line_sep[6]) == 0:
                    state.append(0)
                elif len(line_sep) > 8 and float(line_sep[9]) == 0:
                    state.append(1)
                else:
                    state.append(-1)
        if line == '[ atoms ]\n':
            atom_section = True
    return start_line, atom_name, np.array(atom_num), state


def determine_connection(miss, swap_name_match, main_name, other_name, df_top, main_state):
    """
    Determines the connectivity of the missing atoms in the topology.

    Parameters
    ----------
    miss : list
        Names of atoms which are not present in the current molecule but are present in the target molecule
    swap_name_match : pandas.DataFrame
        Names for all common atoms in the target molecule and their name in the current molecule
    main_name : str
        resname for the molecule of interest.
    other_name : str
        The resname for the other molecule.
    df_top : pandas.DataFrame
        Connectivity of the atoms in each molecule.
    main_state : list
        Which :math:`lambda` state are each atom in the molecule of interest in the dummy state.

    Returns
    -------
    df : pandas.DataFrame
        A pandas Dataframes containing the following information:

          - The missing atoms
          - The real anchor atom that connects them
          - The atom to be used to determine the angle to place the missing atoms
    """
    align_atom, angle_atom = [], []
    df_select = df_top[df_top['Resname'] == main_name]

    # Seperate into each seperate functional group to be added
    anchor_atoms = []

    for m_atom in miss:
        # Find what atoms the missing atom connects to
        connected_atoms = []
        for a in df_select[df_select['Connect 1'] == m_atom]['Connect 2'].values:
            connected_atoms.append(a)
        for a in df_select[df_select['Connect 2'] == m_atom]['Connect 1'].values:
            connected_atoms.append(a)

        # If the atom connects to non-missing atoms than keep these as anchors
        for a in connected_atoms:
            if a not in miss and a not in anchor_atoms:
                anchor_atoms.append(a)

    # Seperate missing atoms connected to each anchor
    miss_sep, align_atoms, angle_atoms = [], [], []
    for anchor in anchor_atoms:
        print(f'anchor: {anchor}')
        miss_anchor = []

        # Which missing atoms are connected to the anchor
        search = True
        included_atoms = [anchor]
        while search is True:
            found_1 = list(df_select[(df_select['Connect 1'].isin(included_atoms)) & (df_select['Connect 2'].isin(miss))]['Connect 2'].values)  # noqa: E501
            found_2 = list(df_select[(df_select['Connect 2'].isin(included_atoms)) & (df_select['Connect 1'].isin(miss))]['Connect 1'].values)  # noqa: E501
            found_atoms = found_1 + found_2
            if len(found_atoms) == 0:
                search = False
            else:
                for atom in found_atoms:
                    miss_anchor.append(atom)
                    included_atoms.append(atom)
                    miss.remove(atom)
        included_atoms.remove(anchor)
        miss_sep.append(included_atoms)
        print(f'miss: {included_atoms}')

        # Find atoms connected to the anchor which are real in main state, but dummy when the atoms we are building are real  # noqa: E501
        align_1 = list(df_select[(df_select['Connect 1'] == anchor) & (df_select['State 2'] != main_state) & (df_select['State 2'] != -1)]['Connect 2'].values)  # noqa: E501
        align_2 = list(df_select[(df_select['Connect 2'] == anchor) & (df_select['State 1'] != main_state) & (df_select['State 1'] != -1)]['Connect 1'].values)  # noqa: E501
        align_atom = align_1 + align_2
        align_atoms.append(align_atom[0])

        # Find the atom to use for matching the angle to ensure that the dummy atom is added in the correct orientation
        ignore_atoms = anchor_atoms + included_atoms + align_atom
        angle_atom_1 = list(df_select[(df_select['Connect 1'] == anchor) & (~df_select['Connect 2'].isin(ignore_atoms))]['Connect 2'].values)  # noqa: E501
        angle_atom_2 = list(df_select[(df_select['Connect 2'] == anchor) & (~df_select['Connect 1'].isin(ignore_atoms))]['Connect 1'].values)  # noqa: E501
        angle_atom = angle_atom_1 + angle_atom_2
        angle_atoms.append(angle_atom[-1])

    # Now let's figure out what these atoms are called in the other molecule
    anchor_atoms_B = _swap_name(anchor_atoms, other_name, swap_name_match)
    angle_atoms_B = _swap_name(angle_atoms, other_name, swap_name_match)
    align_atoms_B = _swap_name(align_atoms, other_name, swap_name_match)
    df = pd.DataFrame(
        {
            'Swap A': main_name,
            'Swap B': other_name,
            'Anchor Atom Name A': anchor_atoms,
            'Anchor Atom Name B': anchor_atoms_B,
            'Alignment Atom A': align_atoms,
            'Alignment Atom B': align_atoms_B,
            'Angle Atom A': angle_atoms,
            'Angle Atom B': angle_atoms_B
        }
    )
    miss_sep_reformat = []
    for missing in miss_sep:
        missing_reformat = missing[0]
        for atom in missing[1:]:
            missing_reformat += f' {atom}'
        miss_sep_reformat.append(missing_reformat)
    df['Missing Atom Name'] = miss_sep_reformat

    return df


def _read_gro(side, resname_list, gro_list):
    """

    """
    name, num = [], []
    in_res = False
    [sim_num, l] = side
    resname = resname_list[sim_num]
    gro = open(gro_list[sim_num]).readlines()
    for line in gro:
        split_line = line.split(' ')
        while '' in split_line:
            split_line.remove('')
        if resname in split_line[0]:
            in_res = True
        elif in_res is True:
            break

        if in_res is True:
            name.append(split_line[1])
            num.append(split_line[2])

    return name, num


def create_atom_map(gro_list, resname_list, swap_patterns):
    """
    If you generate your hybrid topologies in a way that the
    same atom has the same name in each molecule then this
    function will create the atom name transformation map.

    Parameters
    ----------
    gro_list : list
        list of input gro file names
    resname_list : list
        list of residue names with the transformation
    swap_patterns : list of list of intergers
        swapping pattern between simulations

    Returns
    -------
    None just creates pandas DataFrame atom_name_mapping.csv
    """
    output_df = pd.DataFrame()
    for swap_pattern in swap_patterns:
        nameA, numA = _read_gro(swap_pattern[0], resname_list, gro_list)
        nameB, numB = _read_gro(swap_pattern[1], resname_list, gro_list)

        atomnameA, atomidA, atomnameB, atomidB = [], [], [], []
        for n, name in enumerate(nameA):
            if name in nameB:
                atomnameA.append(name)
                atomidA.append(numA[n])
                nb = nameB.index(name)
                atomnameB.append(name)
                atomidB.append(numB[nb])
            element, num, extra = _sep_num_element(name)
            if f'{element}{num}' in nameB:
                atomnameA.append(name)
                atomidA.append(numA[n])
                nb = nameB.index(f'{element}{num}')
                atomnameB.append(f'{element}{num}')
                atomidB.append(numB[nb])
            elif f'{element}V{num}' in nameB:
                atomnameA.append(name)
                atomidA.append(numA[n])
                nb = nameB.index(f'{element}V{num}')
                atomnameB.append(f'{element}V{num}')
                atomidB.append(numB[nb])
            elif f'D{element}{num}' in nameB:
                atomnameA.append(name)
                atomidA.append(numA[n])
                nb = nameB.index(f'D{element}{num}')
                atomnameB.append(f'D{element}{num}')
                atomidB.append(numB[nb])

        df = pd.DataFrame({'resname A': resname_list[swap_pattern[0][0]], 'resname B': resname_list[swap_pattern[1][0]], 'atomid A': atomidA, 'atom name A': atomnameA, 'atomid B': atomidB, 'atom name B': atomnameB})  # noqa: E501
        output_df = pd.concat([output_df, df])
    output_df.reindex()
    output_df.to_csv('atom_name_mapping.csv')
