proj_file = 'project.json'

axes_xyz = ['x', 'y', 'z']

baseline_features = ['jaw'] + \
    [f'center_{x}_{d}' for d in range(3) for x in range(1, 17)]
baseline_y_columns = [f'origin_{x}_{d}' for d in range(3) for x in range(1, 17)]

