"""
Shared logic for reading affine transforms and applying them to images.

This module centralizes the exact logic used by the SimpleReg GUI
(`simplereg.gui.window.RegistrationApp`) so that transformations produced
interactively can be re-applied identically from a script or from the
command line (see `simplereg.apply`).

Benjamin De Leener
Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import re

import numpy as np
import scipy.ndimage as ndimage

from .image import Image

# Interpolation modes accepted by `resample_image_on_grid` / `Image.get_values`.
INTERPOLATION_NEAREST = 0
INTERPOLATION_LINEAR = 1
INTERPOLATION_SPLINE = 3

# Mapping from the CLI/user-facing interpolation names to internal modes.
# 'label' is handled separately (see `apply_label_transform`).
INTERPOLATION_MODES = {
    'nn': INTERPOLATION_NEAREST,
    'linear': INTERPOLATION_LINEAR,
    'spline': INTERPOLATION_SPLINE,
}


def _parse_itk_affine_transform_text(file_text):
    """
    Parse an ITK 'AffineTransform_double_3_3' text transform.

    Per ITK convention, this transform maps points from the FIXED physical space to
    the MOVING physical space (i.e. TransformPoint(p) = A @ (p - c) + t + c, with
    p a point in fixed space). This is the direction expected/produced by ITK-based
    tools such as antsApplyTransforms / sct_apply_transfo.

    :return: 4x4 numpy array mapping fixed RAS points to moving RAS points, or None
             if the file doesn't contain the expected fields.
    """
    params_match = re.search(r"^\s*Parameters\s*:\s*(.+)$", file_text, flags=re.MULTILINE)
    fixed_params_match = re.search(r"^\s*FixedParameters\s*:\s*(.+)$", file_text, flags=re.MULTILINE)
    if params_match is None or fixed_params_match is None:
        return None

    params = np.fromstring(params_match.group(1), sep=' ', dtype=np.float64)
    fixed_params = np.fromstring(fixed_params_match.group(1), sep=' ', dtype=np.float64)
    if params.size != 12 or fixed_params.size != 3:
        raise ValueError("Invalid ITK affine transform format.")

    a = params[:9].reshape((3, 3))
    t = params[9:12]
    c = fixed_params
    offset = t + c - (a @ c)

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = a
    affine_lps[:3, 3] = offset

    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    return ras_to_lps @ affine_lps @ ras_to_lps


def write_itk_affine_transform(fname_affine, affine_matrix, center_phys):
    """
    Write a 4x4 affine transform (RAS, mapping MOVING points to FIXED points, i.e.
    SimpleReg's internal convention) to an ITK 'AffineTransform_double_3_3' text file.

    ITK-based tools (antsApplyTransforms, sct_apply_transfo) expect the transform to
    map FIXED physical points to MOVING physical points, so the matrix is inverted
    before being written. The translation is also re-parameterized relative to
    `center_phys` so that reading the file back (via `_parse_itk_affine_transform_text`,
    which applies ITK's center-relative formula) reproduces the exact same transform.

    :param fname_affine: output file path.
    :param affine_matrix: 4x4 matrix mapping moving RAS points to fixed RAS points.
    :param center_phys: RAS physical point (e.g. moving image barycenter) used as the
                         ITK rotation center.
    """
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    # Invert: SimpleReg's `affine_matrix` maps moving -> fixed, but the ITK file must
    # store the fixed -> moving mapping (see `_parse_itk_affine_transform_text`).
    affine_matrix_inv = np.linalg.inv(np.asarray(affine_matrix, dtype=np.float64))
    affine_itk = ras_to_lps @ affine_matrix_inv @ ras_to_lps

    rotation_matrix = affine_itk[:3, :3]
    offset = affine_itk[:3, 3]

    center = np.asarray(center_phys, dtype=np.float64)
    center_itk = np.array([-center[0], -center[1], center[2]], dtype=np.float64)

    # ITK parameterizes translation relative to a center: offset = t + c - A@c
    translation_array = (offset - center_itk + rotation_matrix @ center_itk).reshape(1, 3)

    with open(fname_affine, 'w') as text_file:
        text_file.write("#Insight Transform File V1.0\n")
        text_file.write("#Transform 0\n")
        text_file.write("Transform: AffineTransform_double_3_3\n")
        text_file.write("Parameters: %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (
            rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2],
            translation_array[0, 0], translation_array[0, 1], translation_array[0, 2]))
        text_file.write("FixedParameters: %.9f %.9f %.9f\n" % (center_itk[0], center_itk[1], center_itk[2]))


def read_transform_from_file(transform_path):
    """
    Read a 4x4 affine transform from file.

    Supported formats: .npy (4x4 matrix), ITK affine transform text (.txt/.tfm/.mat),
    or a plain text file containing a 4x4 or 3x4 matrix.

    :param transform_path: path to the transform file.
    :return: 4x4 numpy array.
    """
    import os

    ext = os.path.splitext(transform_path)[1].lower()

    if ext == '.npy':
        matrix = np.asarray(np.load(transform_path), dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(".npy transform must be a 4x4 matrix.")
        return matrix

    with open(transform_path, 'r', encoding='utf-8') as fobj:
        file_text = fobj.read()

    if "AffineTransform_double_3_3" in file_text:
        matrix = _parse_itk_affine_transform_text(file_text)
        if matrix is not None:
            # `_parse_itk_affine_transform_text` returns the fixed->moving mapping (ITK
            # convention). Invert it back to SimpleReg's internal moving->fixed convention.
            return np.linalg.inv(matrix)

    numeric_tokens = []
    for raw_line in file_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            continue
        numeric_tokens.extend(line.replace(',', ' ').split())

    numeric_values = np.array([float(token) for token in numeric_tokens], dtype=np.float64)

    if numeric_values.size == 16:
        return numeric_values.reshape((4, 4))
    if numeric_values.size == 12:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :] = numeric_values.reshape((3, 4))
        return matrix

    raise ValueError("Unsupported transform format. Expected ITK .txt, 4x4 text matrix, 3x4 text matrix, or .npy.")


def resample_image_on_grid(fixed_img, moving_img, affine_matrix, interpolation_mode=INTERPOLATION_LINEAR,
                            border='constant'):
    """
    Resample `moving_img` onto the voxel grid of `fixed_img`, using `affine_matrix`
    to map physical coordinates from the fixed space to the moving space.

    This is the exact resampling logic used by the SimpleReg GUI when applying
    (and saving) a transform to the moving image.

    :param fixed_img: `Image` defining the output grid.
    :param moving_img: `Image` to resample.
    :param affine_matrix: 4x4 matrix mapping moving physical space to fixed physical space.
    :param interpolation_mode: 0=nearest neighbor, 1=linear, 3=spline (see `Image.get_values`).
    :param border: how to handle points outside the moving image (passed to `Image.get_values`).
    :return: `Image` resampled on the fixed grid.
    """
    nx, ny, nz, _, _, _, _, _ = fixed_img.dim
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
    indexes_ref = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())), dtype=np.float64)
    fixed_phys = fixed_img.transfo_pix2phys(indexes_ref)

    homogeneous = np.hstack([fixed_phys, np.ones((fixed_phys.shape[0], 1), dtype=np.float64)])
    moving_phys = (np.linalg.inv(affine_matrix) @ homogeneous.T).T[:, :3]
    moving_vox = moving_img.transfo_phys2pix(moving_phys, real=False)

    sampled = moving_img.get_values(
        np.array([moving_vox[:, 0], moving_vox[:, 1], moving_vox[:, 2]]),
        interpolation_mode=interpolation_mode,
        border=border
    )

    output = Image(fixed_img)
    if interpolation_mode == INTERPOLATION_NEAREST:
        output.change_type('int32')
    else:
        output.change_type('float32')
    output.data = np.reshape(sampled, (nx, ny, nz))
    return output


def apply_label_transform(fixed_img, moving_img, affine_matrix, dilation_iterations=2):
    """
    Apply a transform to a single-voxel label image (e.g. disc labels, landmarks).

    Classical interpolation may corrupt or erase single-voxel labels, so each
    label is dilated, warped with nearest-neighbour interpolation, and the
    center-of-mass of the resulting blob is extracted to produce a single-voxel
    output label. Not appropriate for multi-voxel labeled segmentations.

    :param fixed_img: `Image` defining the output grid.
    :param moving_img: `Image` containing single-voxel labels.
    :param affine_matrix: 4x4 matrix mapping moving physical space to fixed physical space.
    :param dilation_iterations: number of binary dilation iterations applied to each label
                                 before warping (helps the label survive nearest-neighbour resampling).
    :return: `Image` with single-voxel labels resampled onto the fixed grid.
    """
    label_values = moving_img.getNonZeroValues()

    output = Image(fixed_img)
    output.change_type('float32')
    output.data = np.zeros(fixed_img.data.shape, dtype=np.float32)

    structure = ndimage.generate_binary_structure(3, 1)

    for label_value in label_values:
        label_mask = (moving_img.data == label_value)
        dilated_mask = ndimage.binary_dilation(label_mask, structure=structure, iterations=dilation_iterations)

        dilated_moving_img = Image(moving_img)
        dilated_moving_img.data = dilated_mask.astype(np.float32)

        warped_blob = resample_image_on_grid(
            fixed_img, dilated_moving_img, affine_matrix, interpolation_mode=INTERPOLATION_NEAREST
        )

        blob_mask = warped_blob.data > 0
        if not np.any(blob_mask):
            continue

        center_vox = ndimage.center_of_mass(blob_mask)
        voxel_index = tuple(int(round(coord)) for coord in center_vox)
        if all(0 <= idx < dim for idx, dim in zip(voxel_index, output.data.shape)):
            output.data[voxel_index] = label_value

    return output
