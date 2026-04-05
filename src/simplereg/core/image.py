"""
Heavily inspired from SCT Image API

Benjamin De Leener
Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: Sort out the use of Image.hdr and Image.header --> they seem to carry duplicated information.

import sys
import os
import errno
import itertools
import warnings
import logging
import math
from typing import Sequence, Tuple
from copy import deepcopy
import scipy.ndimage as ndimage
import nibabel as nib

import numpy as np
import pathlib

import transforms3d.affines as affines

logger = logging.getLogger(__name__)

def _get_permutations(im_src_orientation, im_dst_orientation):
    """

    :param im_src_orientation str: Orientation of source image. Example: 'RPI'
    :param im_dest_orientation str: Orientation of destination image. Example: 'SAL'
    :return: list of axes permutations and list of inversions to achieve an orientation change
    """

    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

    perm = [0, 1, 2]
    inversion = [1, 1, 1]
    for i, character in enumerate(im_src_orientation):
        try:
            perm[i] = im_dst_orientation.index(character)
        except ValueError:
            perm[i] = im_dst_orientation.index(opposite_character[character])
            inversion[i] = -1

    return perm, inversion



class Coordinate:
    """
    Class to represent 3D coordinates.

    :param coord: a list with coordinates [x, y, z] or [x, y, z, value] or a string with coordinates delimited by commas.

    Example:
    .. code:: python

        coord = Coordinate([x, y, z])
        coord = Coordinate([x, y, z, value])
    """
    def __init__(self, coord=None):
        if coord is None:
            coord = [0, 0, 0, 0]
        elif isinstance(coord, str):
            coord = [float(c) for c in coord.split(',')]

        if not isinstance(coord, list):
            raise ValueError("Parameter must be a list with coordinates [x, y, z] or [x, y, z, value] or a "
                             "string with coordinates delimited by commas.")

        if len(coord) == 3:
            coord.append(0)
        elif len(coord) != 4:
            raise ValueError("Parameter must be a list with coordinates [x, y, z] or [x, y, z, value].")

        try:
            [float(c) for c in coord]
        except ValueError:
            raise TypeError(f"All coordinates and the value must be float or int: {coord}") from None

        self.x, self.y, self.z, self.value = coord

    def __iter__(self):
        # Allows for this usage: "for x, y, z, v in [list of Coordinate]"
        return iter((self.x, self.y, self.z, self.value))

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z}, {self.value})"

    def __str__(self):
        return f"{self.x},{self.y},{self.z},{self.value}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def permute(self, img_src, orient_dest, mode='absolute'):
        """
        Permute coordinate based on source and destination orientation.

        :param img_src : spinalcordtoolbox.Image() object. Must represent the space that the coordinate
                         is currently in. The source orientation and the dimensions are pulled from
                         this image, which are used to permute/invert the coordinate.
        :param orient_dest: The orientation to output the new coordinate in.
        :param mode: Determines how inversions are handled. If 'absolute', the coordinate is recomputed using
                     a new origin based on the source image's maximum dimension for the inverted axes. If
                     'relative', the coordinate is treated as vector and inverted by multiplying by -1.
        :return: numpy array with the new coordinates in the destination orientation.

        Example:
        .. code:: python

            coord.permute(Image('data.nii.gz'), 'RPI')
        """
        # convert coordinates to array
        coord_arr = np.array([self.x, self.y, self.z])
        dim_arr = np.array(img_src.dim[0:3])
        # permutes
        perm, inversion = _get_permutations(orient_dest, img_src.orientation)  # we need to invert src and dest for this to work
        coord_permute = np.array([coord_arr[perm[0]], coord_arr[perm[1]], coord_arr[perm[2]]])
        dim_permute = np.array([dim_arr[perm[0]], dim_arr[perm[1]], dim_arr[perm[2]]])
        # invert indices based on maximum dimension for each axis
        for i in range(3):
            if inversion[i] == -1:
                if mode == 'absolute':
                    coord_permute[i] = (dim_permute[i] - 1) - coord_permute[i]
                elif mode == 'relative':
                    coord_permute[i] = -1 * coord_permute[i]
                else:
                    raise ValueError(f"Permutations can be 'absolute' or 'relative', but a '{mode}' permutation was "
                                     f"requested instead.")
        return coord_permute

    def __add__(self, other):
        if other == 0:  # this check is necessary for using the function sum() of list
            other = Coordinate()
        return Coordinate([self.x + other.x, self.y + other.y, self.z + other.z, self.value])

    def __radd__(self, other):
        return self + other

    def __truediv__(self, scalar):
        return Coordinate([self.x / float(scalar), self.y / float(scalar), self.z / float(scalar), self.value])



def rpi_slice_to_orig_orientation(dim, orig_orientation, slice_number, axis):
    """
    Convert slice number from RPI (right-posterior-inferior) to original orientation, e.g., AIL
    (anterior-inferior-left).
    :param dim: tuple, dimensions of the image in RPI orientation, e.g., (20, 640, 640).
    :param orig_orientation: str, original image orientation, e.g., AIL.
    :param slice_number: str, slice number in RPI orientation, e.g., 9.
    :param axis: int, axis of the slice in the RPI orientation. '0' corresponds to the x (R-L) axis,
    '1' to the y (A-P) axis, and '2' to the z (I-S) axis.
    :return: int, slice number in original orientation, e.g., 6.

    Example: considering an image with 20 sagittal slices (0-19) and sagittal slice number 13 in the AIL orientation,
    the corresponding slice number in the RPI orientation is 6:
            rpi_slice_to_orig_orientation((20, 640, 640), 'AIL', 13, 0) -> 6
    Note: we use 0 as the last arg in this example as it corresponds to the R-L direction (first axis in RPI)
    """
    # TODO: Consider rewriting this function to use `Coordinate.permute` in much the same way as `reorient_coordinates`.
    #       Since this function basically reimplements the same logic, it's probably overkill.
    # Get the inversions
    _, inversion = _get_permutations('RPI', orig_orientation)

    return (dim[axis] - 1 - slice_number) if inversion[axis] == -1 else slice_number


def reorient_coordinates(coords, img_src, orient_dest, mode='absolute'):
    """
    Reorient coordinates from source image orientation to destination orientation.

    :param coords: iterable of [x, y, z] coordinates to be reoriented
    :param img_src: spinalcordtoolbox.Image() object. Must represent the space that the coordinate
                    is currently in. The source orientation and the dimensions are pulled from
                    this image, which are used to permute/invert the coordinate.
    :param orient_dest: The orientation to output the new coordinate in.
    :param mode: Determines how inversions are handled. If 'absolute', the coordinate is recomputed using
             a new origin based on the source image's maximum dimension for the inverted axes. If
             'relative', the coordinate is treated as vector and inverted by multiplying by -1.
    :return: numpy array with the new coordinates in the destination orientation.
    """

    return [Coordinate(list(coord)).permute(img_src, orient_dest, mode) for coord in coords]


class Slicer(object):
    """
    Provides a sliced view onto original image data.
    Can be used as a sequence.

    Notes:

    - The original image data is directly available without copy,
      which is a nice feature, not a bug! Use .copy() if you need copies...

    Example:

    .. code:: python

       for slice2d in msct_image.SlicerFancy(im3d, "RPI"):
           print(slice)

    """

    def __init__(self, im, orientation="LPI"):
        """

        :param im: image to iterate through
        :param orientation: "from" letters to indicate how to slice the image.
                     The slices are done on the last letter axis,
                     and they are defined as the first/second letter.
        """

        if not isinstance(im, Image):
            raise ValueError("Expecting an image")
        if orientation not in all_refspace_strings():
            raise ValueError("Invalid orientation spec")

        # Get a different view on data, as if we were doing a reorientation

        perm, inversion = _get_permutations(im.orientation, orientation)

        # axes inversion (flip)
        data = im.data[::inversion[0], ::inversion[1], ::inversion[2]]

        # axes manipulations (transpose)
        if perm == [1, 0, 2]:
            data = np.swapaxes(data, 0, 1)
        elif perm == [2, 1, 0]:
            data = np.swapaxes(data, 0, 2)
        elif perm == [0, 2, 1]:
            data = np.swapaxes(data, 1, 2)
        elif perm == [2, 0, 1]:
            data = np.swapaxes(data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
            data = np.swapaxes(data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
        elif perm == [1, 2, 0]:
            data = np.swapaxes(data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
            data = np.swapaxes(data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
        elif perm == [0, 1, 2]:
            # do nothing
            pass
        else:
            raise NotImplementedError()

        self._data = data
        self._orientation = orientation
        self._nb_slices = data.shape[2]

    def __len__(self):
        return self._nb_slices

    def __getitem__(self, idx):
        """

        :return: an image slice, at slicing index idx
        :param idx: slicing index (according to the slicing direction)
        """
        if not isinstance(idx, int):
            raise NotImplementedError()

        if idx >= self._nb_slices:
            raise IndexError("I just have {} slices!".format(self._nb_slices))

        return self._data[:, :, idx]


class SlicerOneAxis(object):
    """
    Image slicer to use when you don't care about the 2D slice orientation,
    and don't want to specify them.
    The slicer will just iterate through the right axis that corresponds to
    its specification.

    Can help getting ranges and slice indices.
    """

    def __init__(self, im, axis="IS"):
        opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
        axis_labels = "LRPAIS"
        if len(axis) != 2:
            raise ValueError()
        if axis[0] not in axis_labels:
            raise ValueError()
        if axis[1] not in axis_labels:
            raise ValueError()
        if axis[0] != opposite_character[axis[1]]:
            raise ValueError()

        for idx_axis in range(2):
            dim_nr = im.orientation.find(axis[idx_axis])
            if dim_nr != -1:
                break
        if dim_nr == -1:
            raise ValueError()

        # SCT convention
        from_dir = im.orientation[dim_nr]
        self.direction = +1 if axis[0] == from_dir else -1
        self.nb_slices = im.dim[dim_nr]
        self.im = im
        self.axis = axis
        self._slice = lambda idx: tuple([(idx if x in axis else slice(None)) for x in im.orientation])

    def __len__(self):
        return self.nb_slices

    def __getitem__(self, idx):
        """

        :return: an image slice, at slicing index idx
        :param idx: slicing index (according to the slicing direction)
        """
        if isinstance(idx, slice):
            raise NotImplementedError()

        if idx >= self.nb_slices:
            raise IndexError("I just have {} slices!".format(self.nb_slices))

        if self.direction == -1:
            idx = self.nb_slices - 1 - idx

        return self.im.data[self._slice(idx)]


class SlicerMany(object):
    """
    Image*s* slicer utility class.

    Can help getting ranges and slice indices.
    Can provide slices (being an *iterator*).

    Use with great care for now, that it's not very documented.
    """

    def __init__(self, images, slicerclass, *args, **kw):
        if len(images) == 0:
            raise ValueError("Don't expect me to work on 0 images!")

        self.slicers = [slicerclass(im, *args, **kw) for im in images]

        nb_slices = [x._nb_slices for x in self.slicers]
        if len(set(nb_slices)) != 1:
            raise ValueError("All images must have the same number of slices along the slicing axis!")
        self._nb_slices = nb_slices[0]

    def __len__(self):
        return self._nb_slices

    def __getitem__(self, idx):
        return [x[idx] for x in self.slicers]


def check_affines_match(im):
    hdr = im.hdr
    hdr2 = hdr.copy()

    try:
        hdr2.set_qform(hdr.get_sform())
    except np.linalg.LinAlgError:
        # See https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3097
        logger.warning("The sform for {} is uninitialized and may cause unexpected behaviour."
                       ''.format(im.absolutepath))

        if im.absolutepath is None:
            logger.error("Internal code has produced an image with an uninitialized sform. "
                         "please report this on github at https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues "
                         "or on the SCT forums https://forum.spinalcordmri.org/.")

        return True

    return np.allclose(hdr.get_qform(), hdr2.get_qform(), atol=1e-3)


class Image(object):
    """
    Create an object that behaves similarly to nibabel's image object. Useful additions include: dim, check_sform and
    a few methods (load, save) that deal with image dtype.
    """

    def __init__(self, param=None, hdr=None, orientation=None, absolutepath=None, dim=None,
                 mmap=(not sys.platform.startswith('win32')), verbose=1, check_sform=False):
        """
        :param param: string indicating a path to a image file or an `Image` object.
        :param hdr: a nibabel header object to use as the header for the image (overwritten if `param` is provided)
        :param orientation: a three character orientation code (e.g. RPI).
        :param absolutepath: a relative path to associate with the image.
        :param dim: The dimensions of the image, defaults to automatically determined.
        :param mmap: Whether to load data arrays as np.memmaps. Defaults to False (i.e. np.array) on Windows, and True
          (i.e. np.memmap) on every other platform. (The context for defaulting to False on Windows is:
          https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3695)
        :param verbose: integer how verbose to be 0 is silent 1 is chatty.
        :param check_sform: whether or not to check whether the sform matches the qform. If this is set to `True`,
          `Image` will fail raise an error if they don't match.
        """

        # initialization of all parameters
        self.affine = None
        self.data = None
        self._path = None
        self.ext = ""

        if absolutepath is not None:
            self._path = os.path.abspath(absolutepath)

        # Case 1: load an image from file
        if isinstance(param, str):
            try:
                self.loadFromPath(param, mmap, verbose)
            except OSError as e:
                if e.errno == errno.EMFILE:
                    e.strerror += (". Please try increasing your system's file descriptor "
                                   "limit by using the command `ulimit -Sn`.")
                raise e
        # Case 2: create a copy of an existing `Image` object
        elif isinstance(param, type(self)):
            self.copy(param)
        # Case 3: create a blank image from a list of dimensions
        elif isinstance(param, list):
            self.data = np.zeros(param)
            self.hdr = hdr.copy() if hdr is not None else nib.Nifti1Header()
            self.hdr.set_data_shape(self.data.shape)
        # Case 4: create an image from an existing data array
        elif isinstance(param, (np.ndarray, np.generic)):
            self.data = param
            self.hdr = hdr.copy() if hdr is not None else nib.Nifti1Header()
            self.hdr.set_data_shape(self.data.shape)
        else:
            raise TypeError('Image constructor takes at least one argument.')

        # Fix any mismatch between the array's datatype and the header datatype
        self.fix_header_dtype()

        # set a more permissive threshold for reading the qform
        # (see https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3703 for details)
        self.hdr.quaternion_threshold = -1e-6

        # Make sure sform and qform are the same.
        # Context: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2429
        if check_sform and not check_affines_match(self):
            logger.error(f"Image {self._path} has different qform and sform matrices. This can produce incorrect "
                         f"results. Please use 'sct_image -i {self._path} -header' to check that both affine "
                         f"matrices are valid. Then, consider running either 'sct_image -set-sform-to-qform' or "
                         f"'sct_image -set-qform-to-sform' to fix any discrepancies you may find.")
            logger.error("If internal SCT code has produced an intermediate/temporary file with this issue, please report this on GitHub at "
                         "https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues or on the SCT forum https://forum.spinalcordmri.org/.")

            # Temporarily skip raising an error, because we now know that "orthogonal qform matrices" from reorientation can cause sform/qform
            # discrepancies that trigger this error.
            #
            # raise ValueError("Image sform does not match qform")
            #
            # Ideally, we would solve the sform/qform discrepancies at the source. But doing so produced an even greater breaking change.
            # So, the safest approach in the short term is to keep the existing results, but skip the above failure and just emit a message.
            # This way, if the above error would trigger, the user at least knows that there may be an issue.
            # Original issue: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4689
            # Secondary issue caused by the "fix" for the first issue: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4744

    @property
    def dim(self):
        return get_dimension(self)

    @property
    def orientation(self):
        return get_orientation(self)

    @property
    def absolutepath(self):
        """
        Storage path (either actual or potential)

        Notes:

        - As several tools perform chdir() it's very important to have absolute paths
        - When set, if relative:

          - If it already existed, it becomes a new basename in the old dirname
          - Else, it becomes absolute (shortcut)

        Usually not directly touched (use `Image.save`), but in some cases it's
        the best way to set it.
        """
        return self._path

    @absolutepath.setter
    def absolutepath(self, value):
        if value is None:
            self._path = None
            return
        elif not os.path.isabs(value) and self._path is not None:
            value = os.path.join(os.path.dirname(self._path), value)
        elif not os.path.isabs(value):
            value = os.path.abspath(value)
        self._path = value

    @property
    def header(self):
        return self.hdr

    @header.setter
    def header(self, value):
        self.hdr = value

    def __deepcopy__(self, memo):
        return type(self)(deepcopy(self.data, memo), deepcopy(self.hdr, memo), deepcopy(self.orientation, memo), deepcopy(self.absolutepath, memo), deepcopy(self.dim, memo))

    def copy(self, image=None):
        if image is not None:
            self.affine = deepcopy(image.affine)
            self.data = deepcopy(image.data)
            self.hdr = deepcopy(image.hdr)
            self._path = deepcopy(image._path)
        else:
            return deepcopy(self)

    def copy_affine_from_ref(self, im_ref):
        """
        Copy qform and sform and associated codes from a reference Image object

        :param im_ref:
        :return:
        """
        # Copy q/sform and code
        self.hdr.set_qform(im_ref.hdr.get_qform())
        self.hdr._structarr['qform_code'] = im_ref.hdr._structarr['qform_code']
        self.hdr.set_sform(im_ref.hdr.get_sform())
        self.hdr._structarr['sform_code'] = im_ref.hdr._structarr['sform_code']

    def set_sform_to_qform(self):
        """Use this (or set_qform_to_sform) when matching matrices are required."""
        self.hdr.set_sform(self.hdr.get_qform())
        self.hdr._structarr['sform_code'] = self.hdr._structarr['qform_code']

    def set_qform_to_sform(self):
        """Use this or (set_sform_to_qform) when matching matrices are required."""
        self.hdr.set_qform(self.hdr.get_sform())
        self.hdr._structarr['qform_code'] = self.hdr._structarr['sform_code']

    def fix_header_dtype(self):
        """
        Change the header dtype to the match the datatype of the array.
        """
        # Using bool for nibabel headers is unsupported, so use uint8 instead:
        # `nibabel.spatialimages.HeaderDataError: data dtype "bool" not supported`
        dtype_data = self.data.dtype
        if dtype_data == bool:
            dtype_data = np.uint8

        dtype_header = self.hdr.get_data_dtype()
        if dtype_header != dtype_data:
            logger.warning(f"Image header specifies datatype '{dtype_header}', but array is of type "
                           f"'{dtype_data}'. Header metadata will be overwritten to use '{dtype_data}'.")
            self.hdr.set_data_dtype(dtype_data)

    def loadFromPath(self, path, mmap, verbose):
        """
        This function load an image from an absolute path using nibabel library

        :param path: path of the file from which the image will be loaded
        :return:
        """

        self.absolutepath = os.path.abspath(path)
        im_file = nib.load(self.absolutepath, mmap=mmap)
        self.affine = im_file.affine.copy()
        self.data = np.asanyarray(im_file.dataobj)
        self.hdr = im_file.header.copy()
        if path != self.absolutepath:
            logger.debug("Loaded %s (%s) orientation %s shape %s", path, self.absolutepath, self.orientation, self.data.shape)
        else:
            logger.debug("Loaded %s orientation %s shape %s", path, self.orientation, self.data.shape)

    def change_shape(self, shape):
        """
        Change data shape (in-place)

        This is mostly useful for adding/removing a fourth dimension,
        you probably don't want to use this function.

        """
        change_shape(self, shape, self)
        return self

    def change_orientation(self, orientation, inverse=False):
        """
        Change orientation on image (in-place).

        :param orientation: orientation string (SCT "from" convention)

        :param inverse: if you think backwards, use this to specify that you actually\
                        want to transform *from* the specified orientation, not *to*\
                        it.

        """
        change_orientation(self, orientation, self, inverse=inverse)
        return self

    def change_type(self, dtype):
        """
        Change data type on image.

        Note: the image path is voided.
        """
        change_type(self, dtype, self)
        return self

    def save(self, path=None, dtype=None, verbose=1, mutable=False):
        """
        Write an image in a nifti file

        :param path: Where to save the data, if None it will be taken from the\
                     absolutepath member.\
                     If path is a directory, will save to a file under this directory\
                     with the basename from the absolutepath member.

        :param dtype: if not set, the image is saved in the same type as input data\
                      if 'minimize', image storage space is minimized\
                        (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                        (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                        (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                        (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                        (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                        (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                        (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                        (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                        (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                        (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                        (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                        (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                        (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
                        (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),

        :param mutable: whether to update members with newly created path or dtype
        """
        if mutable:  # do all modifications in-place
            # Case 1: `path` not specified
            if path is None:
                if self.absolutepath:  # Fallback to the original filepath
                    path = self.absolutepath
                else:
                    raise ValueError("Don't know where to save the image (no absolutepath or path parameter)")
            # Case 2: `path` points to an existing directory
            elif os.path.isdir(path):
                if self.absolutepath:  # Use the original filename, but save to the directory specified by `path`
                    path = os.path.join(os.path.abspath(path), os.path.basename(self.absolutepath))
                else:
                    raise ValueError("Don't know where to save the image (path parameter is dir, but absolutepath is "
                                     "missing)")
            # Case 3: `path` points to a file (or a *nonexistent* directory) so use its value as-is
            #    (We're okay with letting nonexistent directories slip through, because it's difficult to distinguish
            #     between nonexistent directories and nonexistent files. Plus, `nibabel` will catch any further errors.)
            else:
                pass

            if os.path.isfile(path) and verbose:
                logger.warning("File %s already exists. Will overwrite it.", path)
            if os.path.isabs(path):
                logger.debug("Saving image to %s orientation %s shape %s",
                             path, self.orientation, self.data.shape)
            else:
                logger.debug("Saving image to %s (%s) orientation %s shape %s",
                             path, os.path.abspath(path), self.orientation, self.data.shape)

            # Now that `path` has been set and log messages have been written, we can assign it to the image itself
            self.absolutepath = os.path.abspath(path)

            if dtype is not None:
                self.change_type(dtype)

            if self.hdr is not None:
                self.hdr.set_data_shape(self.data.shape)
                self.fix_header_dtype()

            # nb. that copy() is important because if it were a memory map, save() would corrupt it
            dataobj = self.data.copy()
            affine = None
            header = self.hdr.copy() if self.hdr is not None else None
            nib.save(nib.Nifti1Image(dataobj, affine, header), self.absolutepath)
            if not os.path.isfile(self.absolutepath):
                raise RuntimeError(f"Couldn't save image to {self.absolutepath}")
        else:
            # if we're not operating in-place, then make any required modifications on a throw-away copy
            self.copy().save(path, dtype, verbose, mutable=True)
        return self

    def getNonZeroCoordinates(self, sorting=None, reverse_coord=False):
        """
        This function return all the non-zero coordinates that the image contains.
        Coordinate list can also be sorted by x, y, z, or the value with the parameter sorting='x', sorting='y', sorting='z' or sorting='value'
        If reverse_coord is True, coordinate are sorted from larger to smaller.
        """
        n_dim = 1
        if self.dim[3] == 1:
            n_dim = 3
        else:
            n_dim = 4
        if self.dim[2] == 1:
            n_dim = 2

        if n_dim == 3:
            X, Y, Z = (self.data > 0).nonzero()
            list_coordinates = [Coordinate([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
        elif n_dim == 2:
            try:
                X, Y = (self.data > 0).nonzero()
                list_coordinates = [Coordinate([X[i], Y[i], 0, self.data[X[i], Y[i]]]) for i in range(0, len(X))]
            except ValueError:
                X, Y, Z = (self.data > 0).nonzero()
                list_coordinates = [Coordinate([X[i], Y[i], 0, self.data[X[i], Y[i], 0]]) for i in range(0, len(X))]

        if sorting is not None:
            if reverse_coord not in [True, False]:
                raise ValueError('reverse_coord parameter must be a boolean')

            if sorting == 'x':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.x, reverse=reverse_coord)
            elif sorting == 'y':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.y, reverse=reverse_coord)
            elif sorting == 'z':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.z, reverse=reverse_coord)
            elif sorting == 'value':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.value, reverse=reverse_coord)
            else:
                raise ValueError("sorting parameter must be either 'x', 'y', 'z' or 'value'")

        return list_coordinates

    def getNonZeroValues(self, sorting=True):
        """
        This function return all the non-zero unique values that the image contains.
        If sorting is set to True, the list will be sorted.
        """
        list_values = list(np.unique(self.data[self.data > 0]))
        if sorting:
            list_values.sort()
        return list_values

    def getCoordinatesAveragedByValue(self):
        """
        This function computes the mean coordinate of group of labels in the image. This is especially useful for label's images.

        :return: list of coordinates that represent the center of mass of each group of value.
        """
        # 1. Extraction of coordinates from all non-null voxels in the image. Coordinates are sorted by value.
        coordinates = self.getNonZeroCoordinates(sorting='value')

        # 2. Separate all coordinates into groups by value
        groups = dict()
        for coord in coordinates:
            if coord.value in groups:
                groups[coord.value].append(coord)
            else:
                groups[coord.value] = [coord]

        # 3. Compute the center of mass of each group of voxels and write them into the output image
        averaged_coordinates = []
        for value, list_coord in groups.items():
            averaged_coordinates.append(sum(list_coord) / float(len(list_coord)))

        averaged_coordinates = sorted(averaged_coordinates, key=lambda obj: obj.value, reverse=False)
        return averaged_coordinates

    def transfo_pix2phys(self, coordi, mode='absolute'):
        """
        This function returns the physical coordinates of all points of 'coordi'.

        :param coordi: sequence of (nb_points x 3) values containing the pixel coordinate of points.
        :param mode: either 'absolute' or 'relative'.
                     Use 'absolute' to transform absolute pixel coordinates, taking into account
                     the origin of the physical coordinate system. (For example, use 'absolute'
                     for individual voxels.)
                     Use 'relative' to transform relative pixel coordinates, ignoring the origin of
                     the physical coordinate system. (For example, use 'relative' for the
                     difference between two voxels, or for derivatives.)
        :return: sequence with the physical coordinates of the points in the space of the image.

        Example:

        .. code:: python

            img = Image('file.nii.gz')
            coordi_pix = [[1,1,1]]   # for points: (1,1,1). N.B. Important to write [[x,y,z]] instead of [x,y,z]
            coordi_pix = [[1,1,1],[2,2,2],[4,4,4]]   # for points: (1,1,1), (2,2,2) and (4,4,4)
            coordi_phys = img.transfo_pix2phys(coordi=coordi_pix)

        """
        coordi = np.asarray(coordi, dtype=np.float64)
        num_points, dimension = coordi.shape
        if dimension != 3:
            raise ValueError(f'wrong {dimension=}')
        if mode == 'absolute':
            affine_column = np.ones((num_points, 1), dtype=np.float64)
        elif mode == 'relative':
            affine_column = np.zeros((num_points, 1), dtype=np.float64)
        else:
            raise ValueError(f'invalid {mode=}')
        augmented_pix = np.hstack([coordi, affine_column])
        # The affine matrix usually transforms _column_ vectors of pix coordinates, but
        # `augmented_pix` takes the form of _row_ vectors. So, we transpose before and
        # after we do the matrix multiplication.
        affine_matrix = self.hdr.get_best_affine()
        augmented_phys = np.matmul(affine_matrix, augmented_pix.T).T
        return augmented_phys[:, :3]

    def transfo_phys2pix(self, coordi, real=True):
        """
        This function returns the pixels coordinates of all points of 'coordi'

        :param coordi: sequence of (nb_points x 3) values containing the pixel coordinate of points.
        :param real: whether to return real pixel coordinates
        :return: sequence with the physical coordinates of the points in the space of the image.
        """

        m_p2f = self.hdr.get_best_affine()
        m_f2p = np.linalg.inv(m_p2f)
        aug = np.hstack((np.asarray(coordi), np.ones((len(coordi), 1))))
        ret = np.empty_like(coordi, dtype=np.float64)
        for idx_coord, coord in enumerate(aug):
            phys = np.matmul(m_f2p, coord)
            ret[idx_coord] = phys[:3]
        if real:
            return np.int32(np.round(ret))
        else:
            return ret

    def get_values(self, coordi=None, interpolation_mode=0, border='constant', cval=0.0, transform=False):
        """
        This function returns the intensity value of the image at the position coordi (can be a list of coordinates).

        :param coordi: continuouspix
        :param interpolation_mode: 0=nearest neighbor, 1= linear, 2= 2nd-order spline, 3= 2nd-order spline, 4= 2nd-order spline, 5= 5th-order spline
        :return: intensity values at continuouspix with interpolation_mode
        """
        if transform:
            coordi = self.transfo_phys2pix(coordi.T).T
        return ndimage.map_coordinates(self.data, coordi, output=np.float32, order=interpolation_mode,
                                       mode=border, cval=cval)

    def get_transform(self, im_ref, mode='affine'):
        aff_im_self = self.affine
        aff_im_ref = im_ref.affine
        transform = np.matmul(np.linalg.inv(aff_im_self), aff_im_ref)
        if mode == 'affine':
            transform = np.matmul(np.linalg.inv(aff_im_self), aff_im_ref)
        else:
            T_self, R_self, Sc_self, Sh_self = affines.decompose44(aff_im_self)
            T_ref, R_ref, Sc_ref, Sh_ref = affines.decompose44(aff_im_ref)
            if mode == 'translation':
                T_transform = T_ref - T_self
                R_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid':
                T_transform = T_ref - T_self
                R_transform = np.matmul(np.linalg.inv(R_self), R_ref)
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid_scaling':
                T_transform = T_ref - T_self
                R_transform = np.matmul(np.linalg.inv(R_self), R_ref)
                Sc_transform = Sc_ref / Sc_self
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            else:
                transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return transform

    def get_inverse_transform(self, im_ref, mode='affine'):
        aff_im_self = self.affine
        aff_im_ref = im_ref.affine
        if mode == 'affine':
            transform = np.matmul(np.linalg.inv(aff_im_ref), aff_im_self)
        else:
            T_self, R_self, Sc_self, Sh_self = affines.decompose44(aff_im_self)
            T_ref, R_ref, Sc_ref, Sh_ref = affines.decompose44(aff_im_ref)
            if mode == 'translation':
                T_transform = T_self - T_ref
                R_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid':
                T_transform = T_self - T_ref
                R_transform = np.matmul(np.linalg.inv(R_ref), R_self)
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid_scaling':
                T_transform = T_self - T_ref
                R_transform = np.matmul(np.linalg.inv(R_ref), R_self)
                Sc_transform = Sc_self / Sc_ref
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            else:
                transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        return transform

    def get_directions(self):
        """
        This function return the X, Y, and Z axes of the image

        return: X, Y and Z axes of the image
        """
        direction_matrix = self.header.get_best_affine()
        T_self, R_self, Sc_self, Sh_self = affines.decompose44(direction_matrix)
        return R_self[0:3, 0], R_self[0:3, 1], R_self[0:3, 2]

    def interpolate_from_image(self, im_ref, fname_output=None, interpolation_mode=1, border='constant'):
        """
        This function interpolates an image by following the grid of a reference image.
        Example of use:

        .. code:: python

            from spinalcordtoolbox.image import Image
            im_input = Image(fname_input)
            im_ref = Image(fname_ref)
            im_input.interpolate_from_image(im_ref, fname_output, interpolation_mode=1)

        :param im_ref: reference Image that contains the grid on which interpolate.
        :param border: Points outside the boundaries of the input are filled according\
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap')
        :return: a new image that has the same dimensions/grid of the reference image but the data of self image.
        """
        nx, ny, nz, nt, px, py, pz, pt = im_ref.dim
        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
        indexes_ref = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())))
        physical_coordinates_ref = im_ref.transfo_pix2phys(indexes_ref)

        # TODO: add optional transformation from reference space to image space to physical coordinates of ref grid.
        # TODO: add choice to do non-full transorm: translation, (rigid), affine
        # 1. get transformation
        # 2. apply transformation on coordinates

        coord_im = self.transfo_phys2pix(physical_coordinates_ref, real=False)
        interpolated_values = self.get_values(np.array([coord_im[:, 0], coord_im[:, 1], coord_im[:, 2]]), interpolation_mode=interpolation_mode, border=border)

        im_output = Image(im_ref)
        if interpolation_mode == 0:
            im_output.change_type('int32')
        else:
            im_output.change_type('float32')
        im_output.data = np.reshape(interpolated_values, (nx, ny, nz))
        if fname_output is not None:
            im_output.absolutepath = fname_output
            im_output.save()
        return im_output

    def mean(self, dim):
        """
        Average across specified dimension

        :param dim: int: axis used for averaging
        :return: Image object
        """
        im_out = empty_like(self)
        im_out.data = np.mean(self.data, dim)
        return im_out


def concat_data(im_in_list: Sequence[Image], dim, pixdim=None, squeeze_data=False):
    """
    Concatenate data

    :param im_in_list: list of Images
    :param dim: dimension: 0, 1, 2, 3.
    :param pixdim: pixel resolution to join to image header
    :param squeeze_data: bool: if True, remove the last dim if it is a singleton.
    :return im_out: concatenated image
    """
    # WARNING: calling concat_data in python instead of in command line causes a non-understood issue (results are
    # different with both options) from numpy import concatenate, expand_dims

    dat_list = []
    data_concat_list = []

    for i, im in enumerate(im_in_list):
        # if there is more than 100 images to concatenate, then it does it iteratively to avoid memory issue.
        if i != 0 and i % 100 == 0:
            data_concat_list.append(np.concatenate(dat_list, axis=dim))
            dat = im.data
            # if image shape is smaller than asked dim, then expand dim
            if len(dat.shape) <= dim:
                dat = np.expand_dims(dat, dim)
            dat_list = [dat]
            del im
            del dat
        else:
            dat = im.data
            # if image shape is smaller than asked dim, then expand dim
            if len(dat.shape) <= dim:
                dat = np.expand_dims(dat, dim)
            dat_list.append(dat)
            del im
            del dat
    if data_concat_list:
        data_concat_list.append(np.concatenate(dat_list, axis=dim))
        data_concat = np.concatenate(data_concat_list, axis=dim)
    else:
        data_concat = np.concatenate(dat_list, axis=dim)

    im_in_first = im_in_list[0]
    im_out = empty_like(im_in_first)  # NB: empty_like reuses the header from the first input image for im_out
    if im_in_first.absolutepath is not None:
        im_out.absolutepath = add_suffix(im_in_first.absolutepath, '_concat')

    if pixdim is not None:
        im_out.hdr['pixdim'] = pixdim

    if squeeze_data and data_concat.shape[dim] == 1:
        # remove the last dim if it is a singleton.
        im_out.data = data_concat.reshape(
            tuple([x for (idx_shape, x) in enumerate(data_concat.shape) if idx_shape != dim]))
    else:
        im_out.data = data_concat

    # TODO: the line below fails because .dim is immutable. We should find a solution to update dim accordingly
    #  because as of now, this field contains wrong values (in this case, the dimension should be changed). Also
    #  see mean()
    # im_out.dim = im_out.data.shape[:dim] + (1,) + im_out.data.shape[dim:]

    return im_out



def get_dimension(im_file, verbose=1):
    """
    Get dimension from Image or nibabel object. Manages 2D, 3D or 4D images.

    :param: im_file: Image or nibabel object
    :return: nx, ny, nz, nt, px, py, pz, pt
    """
    if not isinstance(im_file, (nib.Nifti1Image, Image)):
        raise TypeError("The provided image file is neither a nibabel.nifti1.Nifti1Image instance nor an Image instance")
    # initializating ndims [nx, ny, nz, nt] and pdims [px, py, pz, pt]
    ndims = [1, 1, 1, 1]
    pdims = [1, 1, 1, 1]
    data_shape = im_file.header.get_data_shape()
    # NB: Nibabel stores dim info as float32. Arithmetic using float32 values can cause issues. So, store the values
    #     as rounded 64-bit float values, according to the spacing of the value.
    zooms = []
    for zoom_f32 in im_file.header.get_zooms():
        # extract the exponent from the spacing used by the float32 values (i.e. 1e-9 -> '-9')
        spacing = np.spacing(zoom_f32)
        spacing_exp = np.floor(np.log10(np.abs(spacing))).astype(int)
        # convert to python float, then round to remove any spurious digits
        zoom_f64 = float(zoom_f32)
        zoom_f64_rounded = round(zoom_f64, -spacing_exp - 1)
        zooms.append(zoom_f64_rounded)
    for i in range(min(len(data_shape), 4)):
        ndims[i] = data_shape[i]
        pdims[i] = zooms[i]
    return *ndims, *pdims


def all_refspace_strings():
    """

    :return: all possible orientation strings ['RAI', 'RAS', 'RPI', 'RPS', ...]
    """
    return [x for x in itertools.chain(*[["".join(x) for x in itertools.product(*seq)] for seq in itertools.permutations(("RL", "AP", "IS"), 3)])]


def get_orientation(im):
    """

    :param im: an Image
    :return: reference space string (ie. what's in Image.orientation)
    """
    res = "".join(nib.orientations.aff2axcodes(im.hdr.get_best_affine()))
    return orientation_string_nib2sct(res)
    return res  # for later ;)


def orientation_string_nib2sct(s):
    """

    :return: SCT reference space code from nibabel one
    """
    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
    return "".join([opposite_character[x] for x in s])


orientation_string_sct2nib = orientation_string_nib2sct


def change_shape(im_src, shape, im_dst=None):
    """

    :param shape: shape to obtain (must be compatible with original one)
    :return: an image with changed shape

    .. note::
        The resulting image has no path
    """

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    if im_src.data.flags.f_contiguous:
        im_dst.data = im_src.data.reshape(shape, order="F")
    elif im_src.data.flags.c_contiguous:
        warnings.warn("Encountered an array with C order, strange!")
        im_dst.data = im_src.data.reshape(shape, order="C")
    else:
        # image data may be a view
        im_dst.data = im_src.data.copy().reshape(shape, order="F")

    pair = nib.nifti1.Nifti1Pair(im_dst.data, im_dst.hdr.get_best_affine(), im_dst.hdr)
    im_dst.hdr = pair.header
    return im_dst


def change_orientation(im_src, orientation, im_dst=None, inverse=False):
    """

    :param im_src: source image
    :param orientation: orientation string (SCT "from" convention)
    :param im_dst: destination image (can be the source image for in-place
                   operation, can be unset to generate one)
    :param inverse: if you think backwards, use this to specify that you actually
                    want to transform *from* the specified orientation, not *to* it.
    :return: an image with changed orientation

    .. note::
        - the resulting image has no path member set
        - if the source image is < 3D, it is reshaped to 3D and the destination is 3D
    """

    if len(im_src.data.shape) < 3:
        pass  # Will reshape to 3D
    elif len(im_src.data.shape) == 3:
        pass  # OK, standard 3D volume
    elif len(im_src.data.shape) == 4:
        pass  # OK, standard 4D volume
    elif len(im_src.data.shape) == 5 and im_src.header.get_intent()[0] == "vector":
        pass  # OK, physical displacement field
    else:
        raise NotImplementedError("Don't know how to change orientation for this image")

    im_src_orientation = im_src.orientation
    im_dst_orientation = orientation
    if inverse:
        im_src_orientation, im_dst_orientation = im_dst_orientation, im_src_orientation

    perm, inversion = _get_permutations(im_src_orientation, im_dst_orientation)

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    im_src_data = im_src.data
    if len(im_src_data.shape) < 3:
        im_src_data = im_src_data.reshape(tuple(list(im_src_data.shape) + ([1] * (3 - len(im_src_data.shape)))))

    # Update data by performing inversions and swaps

    # axes inversion (flip)
    data = im_src_data[::inversion[0], ::inversion[1], ::inversion[2]]

    # axes manipulations (transpose)
    if perm == [1, 0, 2]:
        data = np.swapaxes(data, 0, 1)
    elif perm == [2, 1, 0]:
        data = np.swapaxes(data, 0, 2)
    elif perm == [0, 2, 1]:
        data = np.swapaxes(data, 1, 2)
    elif perm == [2, 0, 1]:
        data = np.swapaxes(data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
        data = np.swapaxes(data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
    elif perm == [1, 2, 0]:
        data = np.swapaxes(data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
        data = np.swapaxes(data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
    elif perm == [0, 1, 2]:
        # do nothing
        pass
    else:
        raise NotImplementedError()

    # Update header

    im_src_aff = im_src.hdr.get_best_affine()
    aff = nib.orientations.inv_ornt_aff(
        np.array((perm, inversion)).T,
        im_src_data.shape)
    im_dst_aff = np.matmul(im_src_aff, aff)

    im_dst.header.set_qform(im_dst_aff)
    im_dst.header.set_sform(im_dst_aff)
    im_dst.header.set_data_shape(data.shape)
    im_dst.data = data

    return im_dst


def change_type(im_src, dtype, im_dst=None):
    """
    Change the voxel type of the image

    :param dtype:    if not set, the image is saved in standard type\
                    if 'minimize', image space is minimize\
                    if 'minimize_int', image space is minimize and values are approximated to integers\
                    (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                    (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                    (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                    (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                    (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                    (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                    (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                    (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                    (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                    (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                    (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                    (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                    (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
                    (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),
    :return:
    """

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    if dtype is None:
        return im_dst

    # get min/max from input image
    min_in = np.nanmin(im_src.data)
    max_in = np.nanmax(im_src.data)

    # find optimum type for the input image
    if dtype in ('minimize', 'minimize_int'):
        # warning: does not take intensity resolution into account, neither complex voxels

        # check if voxel values are real or integer
        isInteger = True
        if dtype == 'minimize':
            for vox in im_src.data.flatten():
                if int(vox) != vox:
                    isInteger = False
                    break

        if isInteger:
            if min_in >= 0:  # unsigned
                if max_in <= np.iinfo(np.uint8).max:
                    dtype = np.uint8
                elif max_in <= np.iinfo(np.uint16):
                    dtype = np.uint16
                elif max_in <= np.iinfo(np.uint32).max:
                    dtype = np.uint32
                elif max_in <= np.iinfo(np.uint64).max:
                    dtype = np.uint64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
            else:
                if max_in <= np.iinfo(np.int8).max and min_in >= np.iinfo(np.int8).min:
                    dtype = np.int8
                elif max_in <= np.iinfo(np.int16).max and min_in >= np.iinfo(np.int16).min:
                    dtype = np.int16
                elif max_in <= np.iinfo(np.int32).max and min_in >= np.iinfo(np.int32).min:
                    dtype = np.int32
                elif max_in <= np.iinfo(np.int64).max and min_in >= np.iinfo(np.int64).min:
                    dtype = np.int64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
        else:
            # if max_in <= np.finfo(np.float16).max and min_in >= np.finfo(np.float16).min:
            #    type = 'np.float16' # not supported by nibabel
            if max_in <= np.finfo(np.float32).max and min_in >= np.finfo(np.float32).min:
                dtype = np.float32
            elif max_in <= np.finfo(np.float64).max and min_in >= np.finfo(np.float64).min:
                dtype = np.float64

        dtype = to_dtype(dtype)
    else:
        dtype = to_dtype(dtype)

        # if output type is int, check if it needs intensity rescaling
        if "int" in dtype.name:
            # get min/max from output type
            min_out = np.iinfo(dtype).min
            max_out = np.iinfo(dtype).max
            # before rescaling, check if there would be an intensity overflow

            if (min_in < min_out) or (max_in > max_out):
                # This condition is important for binary images since we do not want to scale them
                logger.warning(f"To avoid intensity overflow due to convertion to +{dtype.name}+, intensity will be rescaled to the maximum quantization scale")
                # rescale intensity
                data_rescaled = im_src.data * (max_out - min_out) / (max_in - min_in)
                im_dst.data = data_rescaled - (data_rescaled.min() - min_out)

    # change type of data in both numpy array and nifti header
    im_dst.data = getattr(np, dtype.name)(im_dst.data)
    im_dst.hdr.set_data_dtype(dtype)
    return im_dst


def to_dtype(dtype):
    """
    Take a dtypeification and return an np.dtype

    :param dtype: dtypeification (string or np.dtype or None are supported for now)
    :return: dtype or None
    """
    # TODO add more or filter on things supported by nibabel

    if dtype is None:
        return None
    if isinstance(dtype, type):
        if isinstance(dtype(0).dtype, np.dtype):
            return dtype(0).dtype
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, str):
        return np.dtype(dtype)

    raise TypeError("data type {}: {} not understood".format(dtype.__class__, dtype))


def zeros_like(img, dtype=None):
    """

    :param img: reference image
    :param dtype: desired data type (optional)
    :return: an Image with the same shape and header, filled with zeros

    Similar to numpy.zeros_like(), the goal of the function is to show the developer's
    intent and avoid doing a copy, which is slower than initialization with a constant.

    """
    zimg = Image(np.zeros_like(img.data), hdr=img.hdr.copy())
    if dtype is not None:
        zimg.change_type(dtype)
    return zimg


def empty_like(img, dtype=None):
    """
    :param img: reference image
    :param dtype: desired data type (optional)
    :return: an Image with the same shape and header, whose data is uninitialized

    Similar to numpy.empty_like(), the goal of the function is to show the developer's
    intent and avoid touching the allocated memory, because it will be written to
    afterwards.

    """
    dst = change_type(img, dtype)
    return dst


def spatial_crop(im_src, spec, im_dst=None):
    """
    Crop an image in {0,1,2} dimension(s),
    properly altering the header to not change the physical-logical corresondance.

    :param spec: dict of dim -> [lo,hi] bounds (integer voxel coordinates)
    """

    # Compute bounds
    bounds = [(0, x - 1) for x in im_src.data.shape]
    for k, v in spec.items():
        bounds[k] = v

    bounds_ndslice = tuple([slice(a, b + 1) for (a, b) in bounds])

    bounds = np.array(bounds)

    # Crop data
    new_data = im_src.data[bounds_ndslice]

    # Update header
    #
    # Ref: https://mail.python.org/pipermail/neuroimaging/2017-August/001501.html
    # Given A, we want to find A' that is identical up to the intercept, such
    # that A * [x_0, y_0, z_0, 1]' == A' * [0, 0, 0, 1].
    # Conveniently, A' * [0, 0, 0, 1]' is the fourth row in the affine matrix, so
    # we're done as soon as we calculate the LHS:

    aff = im_src.header.get_best_affine()
    new_aff = aff.copy()
    new_aff[:, [3]] = aff.dot(np.vstack((bounds[:, [0]], [1])))

    new_img = nib.Nifti1Image(new_data, new_aff, im_src.header)

    if im_dst is None:
        im_dst = im_src.copy()

    im_dst.header = new_img.header
    im_dst.data = new_data

    return im_dst


def convert(img: Image, squeeze_data=True, dtype=None):
    """
    """
    if squeeze_data:
        img.data = np.squeeze(img.data)
    if dtype:
        img.change_type(dtype)
    return img


def split_img_data(src_img: Image, dim, squeeze_data=True):
    """
    Split data

    :param src_img: input image.
    :param dim: dimension: 0, 1, 2, 3.
    :return: list of split images
    """

    dim_list = ['x', 'y', 'z', 't']
    data = src_img.data

    # in case input volume is 3d and dim=t, create new axis
    if dim + 1 > len(np.shape(data)):
        data = data[..., np.newaxis]

    # in case splitting along the last dim, make sure to remove the last dim to avoid singleton
    if dim + 1 == len(np.shape(data)):
        if squeeze_data:
            do_reshape = True
        else:
            do_reshape = False
    else:
        do_reshape = False

    # Split data into list
    data_split = np.array_split(data, data.shape[dim], dim)

    # Write each file
    im_out_list = []
    for idx_img, dat in enumerate(data_split):
        im_out = empty_like(src_img)
        if do_reshape:
            im_out.data = dat.reshape(tuple([x for (idx_shape, x) in enumerate(data.shape) if idx_shape != dim]))
        else:
            im_out.data = dat
        im_out.absolutepath = add_suffix(src_img.absolutepath, "_{}{}".format(dim_list[dim].upper(), str(idx_img).zfill(4)))
        im_out_list.append(im_out)

    return im_out_list


def concat_warp2d(fname_list, fname_warp3d, fname_dest):
    """
    Concatenate 2d warping fields into a 3d warping field along z dimension. The 3rd dimension of the resulting warping
    field will be zeroed.

    :param fname_list: list of 2d warping fields (along X and Y).
    :param fname_warp3d: output name of 3d warping field
    :param fname_dest: 3d destination file (used to copy header information)
    :return: none
    """

    nx, ny = nib.load(fname_list[0]).shape[0:2]
    nz = len(fname_list)
    warp3d = np.zeros([nx, ny, nz, 1, 3])

    for iz, fname in enumerate(fname_list):
        img = Image(fname)
        warp2d = img.data
        warp3d[:, :, iz, 0, 0] = warp2d[:, :, 0, 0, 0]
        warp3d[:, :, iz, 0, 1] = warp2d[:, :, 0, 0, 1]
        del warp2d

    # save new image
    im_dest = Image(fname_dest)
    im_warp3d = Image(warp3d)
    im_warp3d.copy_affine_from_ref(im_dest)

    # set "intent" code to vector, to be interpreted as warping field
    im_warp3d.header.set_intent('vector', (), '')
    im_warp3d.save(fname_warp3d)


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension.

    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:
    .. code:: python

        add_suffix(t2.nii, _mean) -> t2_mean.nii
        add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return stem + suffix + ext


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    dir_, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            break
    else:
        stem, ext = os.path.splitext(filename)

    return os.path.join(dir_, stem), ext


def check_dim(fname, dim_lst=[3]):
    """
    Check if input dimension matches the input dimension requirements specified in the dim list.
    Example: to check if an image is 2D or 3D: check_dim(my_file, dim_lst=[2, 3])
    :param fname:
    :return: True or False
    """
    dim = Image(fname).hdr['dim'][:4]

    if not dim[0] in dim_lst:
        raise ValueError(f"File {fname} has {dim[0]} dimensions! Accepted dimensions are: {dim_lst}.")


def generate_output_file(fname_in, fname_out, squeeze_data=True, verbose=1):
    """
    Copy fname_in to fname_out with a few convenient checks: make sure input file exists, if fname_out exists send a
    warning, if input and output NIFTI format are different (nii vs. nii.gz) convert by unzipping or zipping, and
    display nice message at the end.
    :param fname_in:
    :param fname_out:
    :param verbose:
    :return: fname_out
    """
    path_in, file_in, ext_in = extract_fname(fname_in)
    path_out, file_out, ext_out = extract_fname(fname_out)

    # create output path (ignore if it already exists)
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

    # if input image does not exist, give error
    if not os.path.isfile(fname_in):
        raise IOError(f"File {fname_in} is not a regular file!")

    # if input and output fnames are the same, do nothing and exit function
    if fname_in == fname_out:
        logger.info("File created: %s", os.path.join(path_out, file_out + ext_out))
        return os.path.join(path_out, file_out + ext_out)

    # if fname_out already exists in nii or nii.gz format
    if os.path.isfile(os.path.join(path_out, file_out + ext_out)):
        logger.warning(f"File {os.path.join(path_out, file_out + ext_out)} already exists. Deleting it..")
        os.remove(os.path.join(path_out, file_out + ext_out))

    if ext_in != ext_out:
        img = Image(fname_in)
        img = convert(img, squeeze_data=squeeze_data)
        img.save(fname_out)
    else:
        # Generate output file without changing the extension
        mv(fname_in, fname_out, verbose=verbose)

    logger.info("File created: %s", os.path.join(path_out, file_out + ext_out))
    return os.path.join(path_out, file_out + ext_out)


def pad_image(im: Image, pad_x_i: int = 0, pad_x_f: int = 0, pad_y_i: int = 0, pad_y_f: int = 0, pad_z_i: int = 0, pad_z_f: int = 0):
    """
    Given an input image, create a copy with specified padding.
    """

    nx, ny, nz, nt, px, py, pz, pt = im.dim
    pad_x_i, pad_x_f, pad_y_i, pad_y_f, pad_z_i, pad_z_f = int(pad_x_i), int(pad_x_f), int(pad_y_i), int(pad_y_f), int(pad_z_i), int(pad_z_f)

    if len(im.data.shape) == 2:
        new_shape = list(im.data.shape)
        new_shape.append(1)
        im.data = im.data.reshape(new_shape)

    # initialize padded_data, with same type as im.data
    padded_data = np.zeros((nx + pad_x_i + pad_x_f, ny + pad_y_i + pad_y_f, nz + pad_z_i + pad_z_f), dtype=im.data.dtype)

    if pad_x_f == 0:
        pad_x_f = None
    elif pad_x_f > 0:
        pad_x_f *= -1
    if pad_y_f == 0:
        pad_y_f = None
    elif pad_y_f > 0:
        pad_y_f *= -1
    if pad_z_f == 0:
        pad_z_f = None
    elif pad_z_f > 0:
        pad_z_f *= -1

    padded_data[pad_x_i:pad_x_f, pad_y_i:pad_y_f, pad_z_i:pad_z_f] = im.data
    im_out = im.copy()
    # TODO: Do not copy the Image(), because the dim field and hdr.get_data_shape() will not be updated properly.
    #   better to just create a new Image() from scratch.
    im_out.data = padded_data  # done after the call of the function
    im_out.absolutepath = add_suffix(im_out.absolutepath, "_pad")

    # adapt the origin in the sform and qform matrix
    new_origin = np.dot(im_out.hdr.get_qform(), [-pad_x_i, -pad_y_i, -pad_z_i, 1])

    im_out.hdr.structarr['qoffset_x'] = new_origin[0]
    im_out.hdr.structarr['qoffset_y'] = new_origin[1]
    im_out.hdr.structarr['qoffset_z'] = new_origin[2]
    im_out.hdr.structarr['srow_x'][-1] = new_origin[0]
    im_out.hdr.structarr['srow_y'][-1] = new_origin[1]
    im_out.hdr.structarr['srow_z'][-1] = new_origin[2]

    return im_out
