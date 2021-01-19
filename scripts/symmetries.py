import numpy as np

def get_symms_square_lattice(system_edge_length, translations=True, point_group=True):
    if translations:
        shifts = range(system_edge_length)
    else:
        shifts = [0]

    if point_group:
        group_ops = []
        group_ops.append([[1, 0], [0, 1]])
        group_ops.append([[-1, 0], [0, 1]])
        group_ops.append([[1, 0], [0, -1]])
        group_ops.append([[-1, 0], [0, -1]])
        group_ops.append([[0, 1], [1, 0]])
        group_ops.append([[0, 1], [-1, 0]])
        group_ops.append([[0, -1], [1, 0]])
        group_ops.append([[0, -1], [-1, 0]])
    else:
        group_ops = [[[1, 0], [0, 1]]]

    transl = []
    for y_shift in shifts:
        for x_shift in shifts:
            for point_group in group_ops:
                ids = []
                for y_start in range(system_edge_length):
                    for x_start in range(system_edge_length):
                        x = point_group[0][0] * x_start + point_group[0][1] * y_start
                        y = point_group[1][0] * x_start + point_group[1][1] * y_start
                        ids.append(get_id_from_pos_2D_square(system_edge_length, x + x_shift, y + y_shift))
                transl.append(ids)

    return transl

def get_id_from_pos_2D_square(system_edge_length, x, y):
    column = x % system_edge_length
    row = y % system_edge_length

    return row * system_edge_length + column

def get_symms_chain(system_edge_length, translations=True, point_group=True):
    if translations:
        shifts = range(system_edge_length)
    else:
        shifts = [0]

    if point_group:
        group_ops = [1, -1]
    else:
        group_ops = [1]

    transl = []
    for x_shift in shifts:
        for point_group in group_ops:
            ids = []
            for x_start in range(system_edge_length):
                x = point_group * x_start
                ids.append(get_id_from_pos_chain(system_edge_length, x + x_shift))
            transl.append(ids)

    return transl

def get_id_from_pos_chain(system_edge_length, x):
    return x % system_edge_length