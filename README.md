# SimpleReg

SimpleReg is a 3D manual registration application for NIfTI medical images,
with 2D/3D visualization and transformation export.

This program was developed in collaboration with artificial intelligence (AI).

## Dependencies

- Python >= 3.10
- PyQt6
- pyqtgraph
- numpy
- scipy
- nibabel
- scikit-image
- transforms3d
- matplotlib

## Installation

1. Clone the repository and move to its root directory.
   ```bash
   git clone https://github.com/neuropoly/simplereg.git
   cd simplereg
   ```

2. Create and activate a Python environment.
   - venv:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   - conda:
   ```bash
   conda create -n simplereg python=3.10
   conda activate simplereg
   ```

3. Install the package in editable mode.
   ```bash
   python3 -m pip install -e .
   ```

## Usage

Launch the application:

```bash
simplereg
```

Launch with an initial transform (`.txt/.tfm/.mat/.npy`):

```bash
simplereg --initial-transform /path/to/transform.txt
```

By default, the initial transform resets the transformation stack.
To append it to the existing stack:

```bash
simplereg --initial-transform /path/to/transform.txt --append-initial-transform
```

Apply a transform and resample the moving image onto the fixed image grid:

```bash
simplereg_apply \
   -i /path/to/moving.nii.gz \
   -d /path/to/fixed.nii.gz \
   -w /path/to/transform.txt \
   -o /path/to/moving_aligned.nii.gz \
   -x linear
```

Available interpolation methods are `nn`, `linear`, `spline`, and `label`.
The `label` method is intended for single-voxel labels and is not suitable for
multi-voxel segmentations.

The package provides two console entry points:

- `simplereg`: launch the graphical registration application.
- `simplereg_apply`: apply a transform and resample an image from the command line.

## Directory Structure

```text
simpleReg/
├── README.md
└── src/
    └── simplereg/
            ├── __main__.py
            ├── apply.py
            ├── core/
            │   ├── image.py
            │   └── transform.py
            └── gui/
                  ├── panels.py
                  ├── utils.py
                  ├── viewers.py
                  └── window.py
```

## Main Modules

- `src/simplereg/__main__.py`: graphical application entry point (CLI arguments, Qt theme, and main window).
- `src/simplereg/apply.py`: applies a transform to an image from Python or the command line (`simplereg_apply`).
- `src/simplereg/gui/window.py`: registration logic (image management, transformation stack, keyboard/mouse interaction, and export).
- `src/simplereg/gui/viewers.py`: 2D visualization widgets (axial/sagittal/coronal) and the 3D view.
- `src/simplereg/gui/panels.py`: control panel (image selection, intensity levels, opacity, transformation stack, and export actions).
- `src/simplereg/gui/utils.py`: colormap/LUT utilities for display.
- `src/simplereg/core/image.py`: image abstraction (NIfTI loading, orientation, voxel/physical-space conversion, and interpolation).

## Features

- Load NIfTI images (`.nii`, `.nii.gz`) as fixed (reference) and moving (source) images.
- Reorient volumes into a common space for consistent interaction.
- Multiplanar display (axial, sagittal, and coronal) with a synchronized cursor.
- Fixed/moving overlay with opacity control and independent intensity-level adjustment.
- 3D visualization with a cursor, slice planes, and fixed/moving volume bounding boxes.
- Automatic initial alignment of the moving image to the fixed image using the center of mass (CoM).
- Manually manipulate the moving image through translation, rotation, and scaling.
- Transformation stack with history, undo, and full reset.
- Import initial transforms (ITK text, numeric text matrices, and 4x4 `.npy` files).
- Export the current transform in ITK format (`.txt/.tfm`) with RAS/LPS conversion.
- Apply the transform on the fixed image grid and save the aligned image.
- Apply transforms outside the GUI using `simplereg_apply`, with output on the fixed image grid.
- Choose the resampling interpolation (`nn`, `linear`, `spline`, or `label`).

## Keyboard Shortcuts

- `T`: translation mode
- `R`: rotation mode
- `S`: scaling mode
- `Esc`: return to navigation mode
- `Ctrl+O`: load images
- `Ctrl+Z`: remove the last transform
- `V`: show/hide the moving image overlay
- Arrow keys: discrete 1 mm translations (`Shift`: 5 mm)
- `PageUp` / `PageDown`: translations along the superior/inferior axis (1 mm, or 5 mm with `Shift`)

Translation, rotation, and scaling modes can also be used by dragging in the
2D views. Rotation is performed around the fixed image center, and scaling
follows the dominant drag axis.


