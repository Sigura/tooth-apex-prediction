import math
import random

import numpy as np

E = np.eye(4)

def get_translation(translation, matrix = None):
    matrix = matrix if matrix is not None else [[0 for _ in range(4)] for _ in range(4)]

    for c in range(3):
        matrix[c][3] = translation[c]

    return np.array(matrix)

x_flip = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

key_teeth = [
    [['1', '2', '3', '4'], ['32', '31', '30', '29']],
    [['13', '14', '15', '16'], ['20', '19', '18', '17']],
    [['8', '9'], ['25', '24']]
]
center_pairs = [
    ( 6, 0.25), ( 7, 0.5), ( 8, 1), ( 9, 1), (10, 0.5), (11, 0.25),
    (22, 0.25), (23, 0.5), (24, 1), (25, 1), (26, 0.5), (27, 0.25)
]
left_pairs = [
    ( 1, 0.5), ( 2, 0.75), ( 3, 1), ( 4, 0.75), ( 5, 0.5), ( 6, 0.25), ( 7, 0.25),
    (26, 0.25), (27, 0.25), (28, 0.5), (29, 0.75), (30, 1), (31, 0.75), (32, 0.5)
]
right_pairs = [
    (16, 0.5), (15, 0.75), (14, 1), (13, 0.75), (12, 0.5), (11, 0.25), (10, 0.25),
    (23, 0.25), (22, 0.25), (21, 0.5), (20, 0.75), (19, 1), (18, 0.75), (17, 0.5)
]

class NotEnoughPoints(Exception):
    pass


def norm(v):
    len_ = np.linalg.norm(v)
    if np.allclose(len_, 0):
        raise ValueError(f'{v} norm is 0')
    return v/len_


class Transform:
    def __init__(self):
        # Initialize the transformation matrix as an identity matrix
        self.matrix = np.eye(4)

    def RotateWXYZ(self, angle, x, y, z):
        """
        Adds a rotation about an arbitrary axis to the current transformation.

        Parameters:
        angle (float): The rotation angle in degrees.
        x, y, z (float): The components of the axis vector.
        """
        # Convert angle from degrees to radians
        angle_rad = np.radians(angle)

        # Normalize the axis vector
        axis = norm(np.array([x, y, z], dtype=float))

        x, y, z = axis

        # Compute the rotation matrix components
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        t = 1 - c

        # Rotation matrix around arbitrary axis
        R = np.array([
            [t*x*x + c,    t*x*y - s*z, t*x*z + s*y, 0],
            [t*x*y + s*z,  t*y*y + c,   t*y*z - s*x, 0],
            [t*x*z - s*y,  t*y*z + s*x, t*z*z + c,   0],
            [0,            0,           0,           1],
        ])

        # Update the transformation matrix by multiplying with the new rotation
        self.matrix = self.matrix @ R  # Note: '@' is the matrix multiplication operator

    def GetMatrix(self):
        """
        Returns the current transformation matrix.
        """
        return self.matrix



def vtk_rotate(angle, p):
    # from vtkmodules.vtkCommonTransforms import vtkTransform
    # tr = vtkTransform()
    tr = Transform()
    # The angle is in degrees, and (x,y,z) specifies
    # the axis that the rotation will be performed around.
    tr.RotateWXYZ(angle, p[0], p[1], p[2])

    matrix = tr.GetMatrix()

    result = [[0 for _ in range(4)] for _ in range(4)]

    for axis in range(4):
        result[axis][axis] = 1

    for axis in range(3):
        for c in range(3):
            result[axis][c] = matrix[c, axis] # matrix.GetElement(c, axis)

    return result


def apply_tr(meshes, transforms):
    result = []
    for i, mesh in enumerate(meshes):
        if mesh is None:
            result.append(None)
            continue

        new_mesh = mesh.copy()

        for j, tr in enumerate(transforms):
            if tr is None:
                continue


            try:
                new_mesh.apply_transform(tr)
            except BaseException as e:
                print(i, j, e)

        result.append(new_mesh)

    return result


def apply_transformation(points, trs):
    points = np.asarray(points)

    # Ensure the points are in homogeneous coordinates
    result = np.column_stack([points, np.ones(points.shape[0])]).transpose()

    # Apply the transformation
    for matrix in trs:
        result = np.matmul(matrix, result)

    return result.transpose()[:, :3]


def apply_tr_vec(vectors, trs = None, normal=False):
    if trs is None:
        trs = [x_flip]

    if len(vectors) <= 0:
        return []

    if type(vectors) == {}.values().__class__:
        vectors = list(vectors)

    vectors = np.asarray(vectors)

    # Ensure the point is in homogeneous coordinates
    vectors_t = np.column_stack([vectors, np.ones(vectors.shape[0])]).transpose()

    for tr in trs:
        vectors_t = np.matmul(tr, vectors_t)

    if not normal:
        return [xyz[:3] for xyz in vectors_t.transpose()]

    return [
        xyz[:3] \
            if not normal or \
                np.linalg.norm(xyz[:3]) == 0 else norm(xyz[:3]) \
            for xyz in vectors_t.transpose()
    ]


def apply_tr_dict(d, trs = None, normal=False):
    keys = d.keys()
    vals = apply_tr_vec(d.values(), trs, normal)
    result = {idx: val for idx, val in zip(keys, vals)}

    return result


def trs_to_tr(trs):
    if len(trs) <= 0:
        return []

    rev = list(reversed(trs))
    result = rev[0];

    for tr in rev[1:]:
        result = np.matmul(result, tr)

    return result


def flip_idx(teeth):
    result = {
        (str((49 if int(idx) > 16 else 17) - int(idx))): val for idx, val in teeth.items()
    }

    return result


def tr_teeth(teeth, trs = None):
    result = {}
    if trs is None:
        trs = [x_flip]

    for idx, tooth in teeth.items():
        center = tooth

        result[idx] = apply_tr_vec([center], trs)[0]

    return result


def teeth_width_norm(teeth, jaw=None):

    if jaw is None:
        l, _, r = teeth_basis(teeth)

        return np.linalg.norm(l - r)

    key_teeth = [
        [left_pairs[:len(left_pairs)//2], left_pairs[len(left_pairs)//2:]],
        [right_pairs[:len(right_pairs)//2], right_pairs[len(right_pairs)//2:]],
        [center_pairs[:len(center_pairs)//2], center_pairs[len(center_pairs)//2:]],

    ]
    lpairs, rpairs, cpairs  = [batch[jaw] for batch in key_teeth]
    l, _, r = teeth_basis(teeth, lpairs=lpairs, cpairs=cpairs, rpairs=rpairs)

    return np.linalg.norm(l - r)


def anterior(centers, return_first=False):
    tpairs  = [(8,9),(7,10),(6,11),(24,25),(23,26),(22,27)]
    weights = {k: x for k,x in center_pairs}
    center_teeth = [(left_i, right_i) for left_i, right_i in tpairs if str(left_i) in centers and str(right_i) in centers]

    center_teeth = sorted(center_teeth, key=lambda x: (weights[x[0]] + weights[x[1]])/2, reverse=True)

    if len(center_teeth) <= 0:
        raise NotEnoughPoints(f'anterior: center teeth not found {centers.keys()}, {center_teeth}')

    if return_first:
        return (centers[str(center_teeth[0][0])] + centers[str(center_teeth[0][1])])/2

    center = weighted_mean(
        [(centers[str(li)] + centers[str(ri)])/2 for li, ri in center_teeth],
        [(weights[li] + weights[ri])/2 for li, ri in center_teeth],
    )


    return center


def weighted_mean(data, weights, axis=0):
    """
    Compute the weighted mean of a numpy array along axis=0.

    Parameters:
    data (numpy.ndarray): The data array of shape (n_samples, n_features).
    weights (numpy.ndarray): The weights array. Can be of shape (n_samples,) or (n_samples, n_features).

    Returns:
    numpy.ndarray: The weighted mean along axis=0 of shape (n_features,).
    """
    data = np.asarray(data, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    # Ensure weights have the correct shape for broadcasting
    if weights.ndim == 1:
        weights = weights[:, np.newaxis]

    # Compute the weighted sum along axis=0
    weighted_sum = np.sum(data * weights, axis=axis)

    # Compute the sum of weights along axis=0
    sum_of_weights = np.sum(weights, axis=axis)

    # Avoid division by zero
    sum_of_weights = np.where(sum_of_weights == 0, np.nan, sum_of_weights)

    # Compute the weighted mean
    weighted_mean = weighted_sum / sum_of_weights

    return weighted_mean


def teeth_basis(centers, cpairs=None, lpairs=None, rpairs=None):
    if cpairs is None:
        cpairs = center_pairs

    if lpairs is None:
        lpairs = left_pairs

    if rpairs is None:
        rpairs = right_pairs

    center_idxs = [str(idx) for idx, _ in cpairs if str(idx) in centers]
    center_teeth = [centers[idx] for idx in center_idxs]

    if len(center_teeth) <= 0:
        raise NotEnoughPoints(f'teeth_basis: center teeth not found. Keys: {centers.keys()}. Pairs: {cpairs}')

    center = anterior(centers)

    pairs = [(str(left_p[0]), str(right_p[0])) for left_p, right_p in zip(lpairs, rpairs) if str(left_p[0]) in centers and str(right_p[0]) in centers]

    left_teeth = [idx for idx, _ in pairs if idx not in center_idxs]

    if len(left_teeth) <= 0:
        center = anterior(centers, return_first=True)
        left_teeth = [idx for idx, _ in pairs]

    if len(left_teeth) <= 0:
        raise NotEnoughPoints(f'teeth_basis: left teeth not found. Keys: {centers.keys()}. Pairs: {lpairs}')

    lpairs = {str(k): w for k, w in lpairs}
    left = weighted_mean(
        [centers[idx] for idx in left_teeth],
        [lpairs[idx] for idx in left_teeth],
    )

    right_teeth = [idx for _, idx in pairs if idx not in center_idxs]

    if len(right_teeth) <= 0:
        center = anterior(centers, return_first=True)
        right_teeth = [idx for _, idx in pairs]

    if len(right_teeth) <= 0:
        raise NotEnoughPoints(f'teeth_basis: right teeth not found. Keys: {centers.keys()}. Pairs: {rpairs}')

    rpairs = {str(k): w for k, w in rpairs}

    right = weighted_mean(
        [centers[idx] for idx in right_teeth],
        [rpairs[idx] for idx in right_teeth],
    )

    # print(json.dumps(np.asarray([left, center, right]).tolist(), indent=2))

    if np.allclose(center - (left+right)/2, np.zeros(3), atol=0.5):
        warnings.warn('Teeth center is equal to the mean of left and right teeth. This may indicate an issue with the input data.', UserWarning)
        center = anterior(centers, return_first=True)

    return np.asarray([left, center, right])


def teeth_norm(centers):
    normal, _ = fit_plane(np.array(list(centers.values())))

    return normal


def teeth_center(teeth):
    center_teeth = np.array([x for x in teeth.values()])

    return center_teeth.mean(axis=0)


def angle_vect(v1, v2):
    v1_u, v2_u = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    cos = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return math.degrees(np.arccos(cos))


def angle_plane(n, v):
    sin = abs((n * v).sum()) / (n ** 2).sum() ** 0.5 / (v ** 2).sum() ** 0.5

    return math.degrees(np.arcsin(sin))


def norm_rots(left, center, right):

    angle1 = math.degrees(np.arctan((right[1] - center[1])/(right[2] - center[2])))
    angle2 = math.degrees(np.arctan((left[1] - center[1])/(left[2] - center[2])))
    angle1 = (angle1 + angle2)/2
    rot1 = vtk_rotate(-angle1, [1, 0, 0])

    angle2 = math.degrees(np.arctan((right[1] - left[1])/(right[0] - left[0])))
    rot2 = vtk_rotate(angle2, [0, 0, 1])

    return [rot2, rot1]

def symmetry_lines(centers):
    l, c, r = teeth_basis(centers)

    lr_dir   = norm(l - r)
    cent_sym = norm(c - np.mean([l, r], axis=0))

    return lr_dir, cent_sym

def project_points2line(points, line_point, line_vector):
    line_vector = line_vector / np.linalg.norm(line_vector)

    vectors = points - line_point

    projections = np.dot(vectors, line_vector)[:, np.newaxis] * line_vector

    return line_point + projections

def transformation_matrix(normal, point=None, target_normal=None, min_angle=True):
    from scipy.spatial.transform import Rotation as R

    point = point if point is not None else np.zeros(3)

    # Normalize the input normal vector
    normal = np.asarray(normal) / np.linalg.norm(normal)

    # Define the normal vector for the XZ plane
    xz_normal = np.asarray([0, 1, 0]) if target_normal is None else target_normal

    normal = fix_norm(xz_normal, normal) if min_angle else normal

    # Calculate the rotation vector between the input normal and the XZ plane normal
    rotation_vector = norm(np.cross(normal, xz_normal))

    # Calculate the angle between the input normal and the XZ plane normal
    angle = np.arccos(np.dot(normal, xz_normal))

    # Create the rotation matrix
    rm = R.from_rotvec(angle * rotation_vector).as_matrix()

    # Create the translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -point

    # Create the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rm
    # transform_matrix = transform_matrix @ translation_matrix

    return [translation_matrix, transform_matrix]

def normal_jaw(centers, tjaw=0, zaxis=None, rotate=True, fit=False):
    if zaxis is None:
        zaxis = np.asarray([0, 0, 1])

    # Check possibility
    l, c, r = teeth_basis(centers)
    lr_dir   = norm(l - r)
    cent_sym = norm(c - np.mean([l, r], axis=0))

    if cent_sym is None or np.allclose(cent_sym, 0):
        return centers, [], []

    # Center alignment
    center = np.mean(list(centers.values()), axis=0)
    center = project_points2line([center], c, cent_sym)[0]

    c_tr    = E.copy() - get_translation(center)
    centers = apply_tr_dict(centers, [c_tr])
    l, r, c, center = apply_tr_vec([l, r, c, center], [c_tr])

    # Symmetry calculation
    lr_dir   = norm(l - r)
    cent_sym = norm(c - np.mean([l, r], axis=0))

    _, land_tr = triangle2xz(np.asarray([-lr_dir/2, cent_sym, lr_dir/2]))
    centers = apply_tr_dict(centers, [land_tr])
    norm_jaw = [c_tr, land_tr]

    # Try to find exact symmetry where possible
    if len(centers) > 12:
        rot2sym  = fix_symmetry(np.asarray(list(centers.values())), axis=[0, 2])
        norm_jaw.append(rot2sym)
        centers  = apply_tr_dict(centers, [rot2sym])

    # Rotate lower jaw to upper (normalization)
    if rotate and tjaw == 0:
        zrot    = np.array(vtk_rotate(180, zaxis))
        norm_jaw.append(zrot)
        centers = apply_tr_dict(centers, [zrot])

    # Fit transformation (normalization)
    if fit:
        fit_tr = E.copy()
        normal = np.asarray(list(centers.values()))
        normal = normal[:, 0].max() - normal[:, 0].min()

        fit_tr = fit_tr/normal

        norm_jaw.append(fit_tr)
        centers = apply_tr_dict(centers, [fit_tr])

    return centers, norm_jaw, [np.linalg.inv(x) for x in norm_jaw[::-1]]


def normal_rot(normal, centroid=None, target_normal=None):
    target_normal = np.asarray([0, 1, 0]) if target_normal is None else target_normal
    T, R = transformation_matrix(normal, target_normal=target_normal, point=centroid, min_angle=False)

    return [T, R], [np.linalg.inv(R), np.linalg.inv(T)]


def rotation_matrix(axis, theta, point=None):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians, around a given point.
    """

    if point is None:
        point = np.zeros(3)

    # Normalize the rotation axis
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))

    # Implement the Rodrigues rotation formula
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    # Translate the point of rotation to the origin
    translation = np.eye(4)
    translation[:3, 3] = -np.array(point)

    # Translate back to the original location
    inv_translation = np.eye(4)
    inv_translation[:3, 3] = np.array(point)

    # Create a 4x4 rotation matrix
    rot_4x4 = np.eye(4)
    rot_4x4[:3, :3] = rotation

    # Combine the transformations
    transform = inv_translation @ rot_4x4 @ translation

    return transform


def angle_between_vectors(v1, v2, component=None):
    v1_u = norm(v1)
    v2_u = norm(v2)

    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

    angle_rad = np.arccos(dot_product)
    # angle_deg = np.degrees(angle_rad)

    cross = np.cross(v1_u, v2_u)

    component = np.argmax(np.abs(cross)) if component is None else component

    if len(cross.shape) <= 0:
        direction = -np.sign(cross)
    else:
        direction = -np.sign(cross[component])

    if direction == 0:
        direction = 1 if np.allclose(v1_u, v2_u) else -1

    return angle_rad * direction


def align_points(A, B):
    # find centroid of A and B
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # subtract centroids from A and B
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # calculate covariance matrix H
    H = np.dot(A_centered.T, B_centered)

    # calculate SVD of H and find rotation matrix R
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    det = np.linalg.det(np.dot(V, U.T))
    diag = np.array([1, 1, det])
    R = np.dot(np.dot(V, np.diag(diag)), U.T)

    # calculate translation vector t
    t = centroid_B - np.dot(R, centroid_A)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def parabola(x, a, b, c):
    return a * np.abs(x + c)**3 + b


def mirror_2dpoints(points, line):
    points = np.array(points)
    line = np.array(line)

    A = line[1, 1] - line[0, 1]
    B = line[1, 0] - line[0, 0]
    C = line[0, 0]*line[1, 1] - line[1, 0]*line[0, 1]

    D = A*points[:, 0] + B*points[:, 1]

    x_prime = ((B**2 - A**2)*points[:, 0] + 2*A*B*points[:, 1] + 2*C*D - 2*B*D) / (A**2 + B**2)
    y_prime = ((A**2 - B**2)*points[:, 1] + 2*A*B*points[:, 0] - 2*A*C + 2*B*D) / (A**2 + B**2)

    return np.column_stack((x_prime, y_prime))

def fit_parabola(x, y, curve=None, initial_guess=None, maxfev=100):
    from scipy.optimize import curve_fit

    if curve is None:
        curve = parabola

    if initial_guess is None:
        initial_guess = [0.03, 0, 0]

    params, perr, infodict, mesg, ier = curve_fit(curve, x, y, p0=initial_guess, maxfev=maxfev, full_output=True)

    return params, infodict['fvec']

def rotate_2dpoints(points, angle):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Translate the points so that the centroid is at the origin
    translated_points = points - centroid

    # Create the rotation matrix
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    # Rotate the points
    rotated_points = np.dot(translated_points, rot_matrix.T)

    # Translate the points back to their original position
    rotated_points += centroid

    return rotated_points


def sum_curve_dist(angle, points, axis_line, weights, initial_guess, curve, maxfev, mirror, filt, return_coef=False):
    rotated_points  = rotate_2dpoints(points.copy(), angle)
    inf = float('inf')

    if mirror:
        mirrored_points = mirror_2dpoints(rotated_points, axis_line)
        both_points = np.concatenate((rotated_points, mirrored_points))
    else:
        both_points = rotated_points

    coef = []

    try:
        coef, perr = fit_parabola(both_points[:, 0], both_points[:, 1], maxfev=maxfev, curve=curve, initial_guess=initial_guess)
    except RuntimeError as e:
        # print('sum_curve_dist', e)
        if not return_coef:
            return inf
        else:
            return inf, None, None, None

    if not filt(coef):
        if not return_coef:
            return inf
        else:
            return inf, *coef

    result = np.sum(1.2 ** np.abs(perr))

    if not return_coef:
        return result

    return result, *coef


def find_parabola_angle(
    points, axis=None,
    start_angle=-np.pi, end_angle=np.pi,
    steps=180, weights=None,
    maxfev=100, filt=None,
    verbose=False, mirror=True,
    curve=None,
    angle_tol=5,
    initial_guess=None
):
    import math
    # from scipy.optimize import minimize

    if curve is None:
        curve = parabola

    if steps <= 1:
        steps = 2

    if axis is None:
        axis = [0, 1]

    if weights is None:
        weights = np.ones(len(points))

    if filt is None:
        filt = lambda x: True

    if mirror:
        weights = np.concatenate((weights, weights))

    angle_tol = np.radians(angle_tol)
    axis_line = np.asarray([[0, 0], axis])

    counter = 1

    while True:
        current_steps = max(int(steps/counter), 3)
        step   = (end_angle - start_angle)/(current_steps-1)

        angles    = np.linspace(start_angle, end_angle + step, current_steps)
        distances = np.asarray([[angle, *sum_curve_dist(angle, points, axis_line, weights, initial_guess, curve, maxfev, mirror, filt, return_coef=True)] for angle in angles])
        idx    = np.argmin(distances[:, 1])
        row    = distances[idx]

        start_angle = row[0] - step/2
        end_angle   = row[0] + step/2

        counter += 1

        if step < angle_tol:
            break

    if verbose:
        print(idx, row[0], round(np.degrees(row[0]), 3), row[2:], counter)
        print(*['{:02d} {:3.3f}  | {:6.0f} | {:1.5f} {:2.5f} {:2.5f}'.format(i, *x) for i, x in enumerate(distances) if x is not None and not math.isinf(x[1])], sep='\n')

    return row

def triangle2xz(input_points, up=None):
    from scipy.spatial.transform import Rotation as R

    assert len(input_points) == 3, 'must be triangle'

    if up is None:
        up = np.asarray([0, 1, 0])

    centroid = (input_points[0]+input_points[2])/2
    points   = input_points - centroid

    v1 = points[0] - points[1]
    v2 = points[2] - points[1]
    normal = norm(np.cross(v1, v2))

    rot2pl = None
    # normal    = fix_norm(up, normal)
    if np.allclose(normal, up):
        rot2pl    = np.eye(4)
    elif np.allclose(-normal, up):
        z_axis = np.cross(norm(input_points[1] - centroid), norm(input_points[2]-input_points[0]))
        rot2pl    = rotation_matrix(z_axis, theta=np.radians(180))
    else:
        rot2pl, _ = normal_rot(normal, up)
        rot2pl    = rot2pl[1]

    points = np.concatenate((np.zeros((1, 3)), points))

    points = apply_transformation(points, [rot2pl])

    rot_angle = angle_between_vectors(norm(points[2]-points[0])[[0,2]], [0, 1])
    R = rotation_matrix(up, rot_angle, point=points[0])
    points = apply_transformation(points, [R])

    points = points[1:]-points[0]
    zdir   = points[1][:-1]

    # assert np.allclose(zdir, np.zeros(2)), f'wrong direction awaiting Z axis {zdir}'
    assert np.allclose(np.linalg.norm(input_points[1] - input_points[0]), np.linalg.norm(points[1] - points[0])), '1-0 sides must be equal'
    assert np.allclose(np.linalg.norm(input_points[2] - input_points[0]), np.linalg.norm(points[2] - points[0])), '2-0 sides must be equal'
    assert np.allclose(np.linalg.norm(input_points[1] - input_points[2]), np.linalg.norm(points[1] - points[2])), '2-1 sides must be equal'

    tr = align_points(input_points, points)

    return points, tr


def fix_rot(centers):
    lr_dir, cent_sym = symmetry_lines(centers)

    if lr_dir is None:
        raise NotEnoughPoints(f'Could not find symetry line. Teeth count: {len(centers)}/32')

    triangle = np.asarray([-lr_dir/2, cent_sym, lr_dir/2])
    _, land_tr = triangle2xz(triangle)
    centers = apply_tr_dict(centers, [land_tr])

    return land_tr # @ rot2sym


def fit_teeth_tr(centers):
    l, _, r = teeth_basis(centers)

    d = abs(r[0] - l[0])

    return E.copy()/d


def center_teeth_tr(centers):

    centroid = teeth_center(centers)

    return E.copy() - get_translation(centroid)


def min_(a):
    if len(a) <= 0:
        return None

    return a.min()


def max_(a):
    if len(a) <= 0:
        return None

    return a.max()


def max_dist(vers):
    x = vers[:, 0]
    y = vers[:, 1]
    z = vers[:, 2]

    min_x, max_x = min_(x[x > 0]), max_(x[x < 0])
    min_y, max_y = min_(y[y > 0]), max_(y[y < 0])
    min_z, max_z = min_(z[z > 0]), max_(z[z < 0])
    d_x = min_x - max_x if min_x is not None and max_x is not None else 0
    d_y = min_y - max_y if min_y is not None and max_y is not None else 0
    d_z = min_z - max_z if min_z is not None and max_z is not None else 0

    result = {'x': d_x, 'y': d_y, 'z': d_z}

    # print((min_x, max_x), (min_y, max_y), (min_z, max_z), sorted(result.items(), key=lambda r: r[1], reverse=True))

    return sorted(result.items(), key=lambda r: abs(r[1]), reverse=True)[0]


def y_rot(centers):
    pairs = [[8, 9], [7, 10], [6, 11], [25, 24], [26, 23], [27, 22]]
    pairs = [
        [idx1, idx2]  for idx1, idx2 in pairs if str(idx1) in centers and str(idx2) in centers
    ]
    center_teeth = np.array([[centers[str(idx1)], centers[str(idx2)]] for idx1, idx2 in pairs])
    cs = center_teeth.mean(axis=0)
    v = cs[1]-cs[0]
    normal = norm(v)

    x_angle = angle_vect(np.array([1, 0, 0]), normal)
    x_angle = x_angle if abs(x_angle) < 45 else x_angle - 90
    x_rot = vtk_rotate(-x_angle, [0, 1, 0])

    # print(x_angle, normal)

    return x_rot


def get_aug_trs(tr_range=(-0.07, 0.07), angle_range=(-3, 3), with_flip_x=True):

    tr_random = E.copy() + get_translation([random.uniform(*tr_range), random.uniform(*tr_range), random.uniform(*tr_range)])
    tr_rotate_x = vtk_rotate(random.uniform(*angle_range), [1, 0, 0])
    tr_rotate_y = vtk_rotate(random.uniform(*angle_range), [0, 1, 0])
    tr_rotate_z = vtk_rotate(random.uniform(*(np.array(angle_range)/2).tolist()), [0, 0, 1])

    tr_augs = []

    if with_flip_x and bool(random.getrandbits(1)):
        tr_augs.append(x_flip.copy())

    tr_augs += [
        tr_rotate_x,
        tr_rotate_y,
        tr_rotate_z,
        tr_random
    ]

    return tr_augs


def fix_symmetry(centers, axis=None, steps=16, return_angle=False, initial_guess=None, verbose=False):
    if axis is None:
        axis = [0, 2]
    if initial_guess is None:
        initial_guess = [-0.03, 10, 0]
    axises = np.arange(3)
    rot_axis = axises[~np.isin(axises, axis)][0]
    normal   = np.zeros(3)
    normal[rot_axis] = 1
    start_angle = -np.pi/2

    points = centers[:, axis]
    min_c = np.abs([points[:, 0].min(), points[:, 0].max()]).min()/6

    row      = find_parabola_angle(
        points,
        start_angle = start_angle,
        end_angle   = start_angle+2*np.pi, axis=[0, 1],
        steps=steps, filt=lambda x: x[0] < -0.0004 and abs(x[2]) < min_c,
        initial_guess=initial_guess, verbose=verbose
    )

    angle = -row[0]
    rot      = rotation_matrix(normal, angle) #, point=centroid)
    if return_angle:
        return rot, angle
    return rot


def teeth_rot_trs(centers, up=None):
    if up is None:
        up     = np.asarray([0, 1, 0])

    if len(centers) < 4:
        raise NotEnoughPoints(f'Could not find center. Teeth count: {len(centers)}/32')

    cent_tr    = center_teeth_tr(centers)
    centers    = apply_tr_dict(centers, [cent_tr])

    normal, _  = fit_plane(np.asarray(list(centers.values())))
    normal     = fix_norm(up, normal)
    rot2pl, _  = normal_rot(normal, target_normal=up)
    centers    = apply_tr_dict(centers, rot2pl[1:])

    rot2sym    = fix_symmetry(np.asarray(list(centers.values())), axis=[0, 2])
    centers    = apply_tr_dict(centers, [rot2sym])

    return [cent_tr, *rot2pl[1:], rot2sym]


def norm_points(points, jaw=0, to_one=True, up=None, zaxis=None):
    """
    rotate jaw to z direction (apex to Y/UP)
    """
    if up is None:
        up = np.asarray([0, 1, 0])

    if zaxis is None:
        zaxis = np.asarray([0, 0, 1])

    c_tr    = center_teeth_tr(points)
    centers = apply_tr_dict(points, [c_tr])

    fix_rot_tr = fix_rot(centers)
    centers    = apply_tr_dict(centers, [fix_rot_tr])


    fit_tr = E.copy()
    normal = teeth_width_norm(centers, 0)

    if to_one:
        fit_tr = fit_tr/normal
        centers = apply_tr_dict(centers, [fit_tr])

    result = apply_tr_dict(points, [c_tr, fix_rot_tr, fit_tr])

    return result, normal, [c_tr, fix_rot_tr, fit_tr]


def get_teeth_transform(data, jaw = 'maxilla_frames'):
    teeth = {
        idx: tooth for phase in data['phases'] \
            for step in phase[jaw] \
                for idx, tooth in step.items()
                    if jaw in phase and step is not None and tooth is not None
    }

    return teeth


def transition(points, direction=None):
    c = direction if direction is not None else -points.mean(axis=0)

    return np.array([x+c for x in points])

