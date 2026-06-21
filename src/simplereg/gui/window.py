import numpy as np
import pyqtgraph as pg
import os
import re
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QGridLayout, QFileDialog, QMessageBox, QInputDialog)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter

# Imports relatifs ou absolus selon votre configuration PYTHONPATH
# Ici on assume que le script est lancé depuis la racine simpleReg
from core.image import Image, add_suffix
from gui.viewers import SliceWidget, BrainViewer3D
from gui.panels import RegistrationControlPanel
from gui.utils import get_lut_for_colormap


class RegistrationApp(QMainWindow):
    TARGET_RESOLUTION_MM = 1.0  # mm
    CONTROL_PANEL_WIDTH = 350
    BOX_EDGES = (
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BrainThrough - Manual Registration")
        self.resize(1400, 900)

        # Données
        self.images = {}  # {name: {'obj': Image, 'data': numpy}}
        self.fixed_img_name = None
        self.moving_img_name = None
        self.affine_matrix = np.eye(4)
        self.fixed_visible = True
        self.moving_visible = True
        self.fixed_opacity = 1.0
        self.moving_opacity = 0.6
        self.fixed_levels = (0.0, 1.0)
        self.moving_levels = (0.0, 1.0)
        self.interaction_mode = 'navigation'
        # Pile de transformations
        self.transform_stack = []       # [{'matrix': 4x4, 'label': str}, ...]
        self._pending_matrix = None     # accumulateur pendant un drag en cours
        self._pending_meta = {}         # métadonnées pour le label final
        self._scaling_drag_state = {
            'axial': {'start': None, 'axis': None},
            'sagittal': {'start': None, 'axis': None},
            'coronal': {'start': None, 'axis': None},
        }

        # État curseur
        self.cursor_pos = [0.0, 0.0, 0.0]
        self.fixed_shape = (1, 1, 1)

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 1. Panneau (Importé de panels.py)
        self.panel = RegistrationControlPanel(self)
        self.panel.setFixedWidth(self.CONTROL_PANEL_WIDTH)
        self.panel.combo_fixed.currentTextChanged.connect(self.set_fixed)
        self.panel.combo_moving.currentTextChanged.connect(self.set_moving)
        self.panel.btn_load_fixed.clicked.connect(self.load_fixed_image)
        self.panel.btn_load_moving.clicked.connect(self.load_moving_image)
        self.panel.btn_toggle_fixed.toggled.connect(self.toggle_fixed_visibility)
        self.panel.btn_toggle_moving.toggled.connect(self.toggle_moving_visibility)
        self.panel.btn_align_com.clicked.connect(self.align_moving_to_fixed_com)
        self.panel.btn_load_initial_transform.clicked.connect(self.load_initial_transform)
        self.panel.btn_save_transform.clicked.connect(self.save_current_transform_itk)
        self.panel.btn_apply_and_save.clicked.connect(self.apply_transform_to_moving_and_save)
        layout.addWidget(self.panel)

        self.panel.set_fixed_visibility_label(self.fixed_visible)
        self.panel.set_moving_visibility_label(self.moving_visible)

        # 2. Grille de visualisation (Importé de viewers.py)
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(2)

        self.view_axial = SliceWidget("Axial (XY)", 'g')
        self.view_sagittal = SliceWidget("Sagittal (YZ)", 'r')
        self.view_coronal = SliceWidget("Coronal (XZ)", 'b')
        self.view_3d = BrainViewer3D()

        self.view_axial.setup_orientation_labels('P', 'A', 'L', 'R')
        self.view_sagittal.setup_orientation_labels('S', 'I', 'A', 'P')
        self.view_coronal.setup_orientation_labels('S', 'I', 'L', 'R')

        grid.addWidget(self.view_axial, 0, 0)
        grid.addWidget(self.view_sagittal, 0, 1)
        grid.addWidget(self.view_coronal, 1, 0)
        grid.addWidget(self.view_3d, 1, 1)

        layout.addWidget(grid_widget)

        # Signaux navigation
        self.view_axial.sig_clicked.connect(lambda x, y: self.set_cursor(x, y, 0, 1))
        self.view_sagittal.sig_clicked.connect(lambda x, y: self.set_cursor(x, y, 1, 2))
        self.view_coronal.sig_clicked.connect(lambda x, y: self.set_cursor(x, y, 0, 2))
        self.view_axial.sig_drag.connect(
            lambda x, y, dx, dy, is_start, is_finish: self.handle_view_drag(
                'axial', self.view_axial, 2, x, y, dx, dy, is_start, is_finish
            )
        )
        self.view_sagittal.sig_drag.connect(
            lambda x, y, dx, dy, is_start, is_finish: self.handle_view_drag(
                'sagittal', self.view_sagittal, 0, x, y, dx, dy, is_start, is_finish
            )
        )
        self.view_coronal.sig_drag.connect(
            lambda x, y, dx, dy, is_start, is_finish: self.handle_view_drag(
                'coronal', self.view_coronal, 1, x, y, dx, dy, is_start, is_finish
            )
        )

        self.set_interaction_mode('navigation')

        # Menu
        self.create_menu()

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        load_action = file_menu.addAction('Load Image')
        load_action.triggered.connect(self.load_image)
        load_action.setShortcut("Ctrl+O")

    def load_image(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "NIfTI (*.nii *.nii.gz)")
        for fname in fnames:
            self.register_image(fname)

    def register_image(self, fname):
        try:
            img = Image(fname)
            img.change_orientation('RPI')
            data = img.data.astype(np.float32)

            base_name = os.path.basename(fname)
            name = base_name
            suffix = 1
            while name in self.images:
                suffix += 1
                name = f"{base_name} ({suffix})"

            self.images[name] = {'obj': img, 'data': data}
            self.panel.combo_fixed.addItem(name)
            self.panel.combo_moving.addItem(name)

            if self.fixed_img_name is None:
                self.panel.combo_fixed.setCurrentText(name)

            return name
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            return None

    def load_fixed_image(self):
        fixed_path, _ = QFileDialog.getOpenFileName(self, "Open Fixed Image", "", "NIfTI (*.nii *.nii.gz)")
        if not fixed_path:
            return

        fixed_name = self.register_image(fixed_path)
        if fixed_name:
            self.panel.combo_fixed.setCurrentText(fixed_name)

    def load_moving_image(self):
        moving_path, _ = QFileDialog.getOpenFileName(self, "Open Moving Image", "", "NIfTI (*.nii *.nii.gz)")
        if not moving_path:
            return

        moving_name = self.register_image(moving_path)
        if moving_name:
            self.panel.combo_moving.setCurrentText(moving_name)

    def set_fixed(self, name):
        if not name or name not in self.images: return
        self.fixed_img_name = name
        self.fixed_shape = self.images[name]['data'].shape
        self.fixed_levels = self._get_image_default_levels(name)
        self.panel.set_display_level_controls('fixed', *self.fixed_levels)

        self.cursor_pos = [((s - 1) / 2.0) for s in self.fixed_shape]

        center_phys = self._get_fixed_center_phys()
        extent_mm = self._get_fixed_extent_mm()
        self.view_3d.center_of_rotation = center_phys
        self.view_3d.setCameraPosition(distance=max(150.0, 1.5 * np.linalg.norm(extent_mm)), elevation=30, azimuth=45)
        self.view_3d.show_trajectory_planes(True)

        self._reset_slice_views()

        self.update_transform()

    def set_moving(self, name):
        self.moving_img_name = name
        if name and name in self.images:
            self.moving_levels = self._get_image_default_levels(name)
            self.panel.set_display_level_controls('moving', *self.moving_levels)
        self.update_transform()

    def toggle_fixed_visibility(self, checked):
        self.fixed_visible = bool(checked)
        self.panel.set_fixed_visibility_label(self.fixed_visible)
        self.refresh_display()

    def toggle_moving_visibility(self, checked):
        self.moving_visible = bool(checked)
        self.panel.set_moving_visibility_label(self.moving_visible)
        self.refresh_display()

    def update_display_levels(self, key):
        if key.startswith('fixed'):
            self.fixed_levels = self._sanitize_levels(self.panel.get_display_levels('fixed'))
        elif key.startswith('moving'):
            self.moving_levels = self._sanitize_levels(self.panel.get_display_levels('moving'))
        self.refresh_display()

    def update_opacity(self, which, value):
        if which == 'fixed':
            self.fixed_opacity = value
        elif which == 'moving':
            self.moving_opacity = value
        self.refresh_display()

    def set_cursor(self, mx, my, dim_h, dim_v):
        if self._get_fixed_image() is None:
            return

        flip_x = True
        flip_y = False
        if (dim_h, dim_v) in ((1, 2), (0, 2)):
            flip_x = False
            flip_y = False

        pos = np.asarray(self.cursor_pos, dtype=np.float64)
        pos[dim_h] = float(self.fixed_shape[dim_h] - 1 - mx) if flip_x else float(mx)
        pos[dim_v] = float(self.fixed_shape[dim_v] - 1 - my) if flip_y else float(my)
        clipped = np.clip(pos, np.zeros(3, dtype=np.float64), np.array(self.fixed_shape, dtype=np.float64) - 1.0)
        self.cursor_pos = clipped.tolist()
        self.refresh_display()

    def _get_fixed_image(self):
        if not self.fixed_img_name or self.fixed_img_name not in self.images:
            return None
        return self.images[self.fixed_img_name]['obj']

    def _get_moving_image(self):
        if not self.moving_img_name or self.moving_img_name not in self.images:
            return None
        return self.images[self.moving_img_name]['obj']

    def _get_fixed_center_vox(self):
        return (np.array(self.fixed_shape, dtype=np.float64) - 1.0) / 2.0

    def _get_fixed_center_phys(self):
        fixed_img = self._get_fixed_image()
        if fixed_img is None:
            return np.zeros(3, dtype=np.float64)
        return fixed_img.transfo_pix2phys([self._get_fixed_center_vox()])[0]

    def _get_axis_step_vectors(self, image_obj):
        relative_steps = image_obj.transfo_pix2phys(np.eye(3, dtype=np.float64), mode='relative')
        spacings = np.linalg.norm(relative_steps, axis=1)
        directions = np.zeros_like(relative_steps)
        valid = spacings > 0
        directions[valid] = relative_steps[valid] / spacings[valid, np.newaxis]
        directions[~valid] = np.eye(3, dtype=np.float64)[~valid]
        return relative_steps, spacings, directions

    def _get_fixed_spacings(self):
        fixed_img = self._get_fixed_image()
        if fixed_img is None:
            return np.ones(3, dtype=np.float64)
        _, spacings, _ = self._get_axis_step_vectors(fixed_img)
        return spacings

    def _get_fixed_directions(self):
        fixed_img = self._get_fixed_image()
        if fixed_img is None:
            return np.eye(3, dtype=np.float64)
        _, _, directions = self._get_axis_step_vectors(fixed_img)
        return directions

    def _get_fixed_extent_mm(self):
        return np.maximum(np.array(self.fixed_shape, dtype=np.float64) - 1.0, 0.0) * self._get_fixed_spacings()

    def _get_image_box_corners_phys(self, image_obj):
        shape = np.array(image_obj.data.shape[:3], dtype=np.float64)
        max_idx = np.maximum(shape - 1.0, 0.0)
        corners_vox = np.array([
            [0.0, 0.0, 0.0],
            [max_idx[0], 0.0, 0.0],
            [0.0, max_idx[1], 0.0],
            [max_idx[0], max_idx[1], 0.0],
            [0.0, 0.0, max_idx[2]],
            [max_idx[0], 0.0, max_idx[2]],
            [0.0, max_idx[1], max_idx[2]],
            [max_idx[0], max_idx[1], max_idx[2]],
        ], dtype=np.float64)
        return image_obj.transfo_pix2phys(corners_vox)

    def _phys_points_to_segments(self, corners_phys):
        corners = np.asarray(corners_phys, dtype=np.float64)
        return np.asarray([(corners[i], corners[j]) for i, j in self.BOX_EDGES], dtype=np.float64)

    def _transform_phys_points(self, points_phys, transform):
        points = np.asarray(points_phys, dtype=np.float64)
        homog = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
        return (transform @ homog.T).T[:, :3]

    def _project_phys_segments_to_view(self, segments_phys, spec):
        fixed_img = self._get_fixed_image()
        if fixed_img is None or segments_phys is None:
            return np.empty((0, 2, 2), dtype=np.float64)

        seg_pts_phys = np.asarray(segments_phys, dtype=np.float64).reshape(-1, 3)
        seg_pts_vox = fixed_img.transfo_phys2pix(seg_pts_phys, real=True)

        delta = seg_pts_vox - spec['p_center_vox']
        u = delta @ spec['vec_h_vox']
        v = delta @ spec['vec_v_vox']
        u += (spec['plane_size_h'] - 1.0) / 2.0
        v += (spec['plane_size_v'] - 1.0) / 2.0

        return np.stack([u, v], axis=1).reshape(-1, 2, 2)

    def _resolution_from_extent_mm(self, extent_mm, target_step_mm=None):
        if target_step_mm is None:
            target_step_mm = self.TARGET_RESOLUTION_MM
        extent = max(0.0, float(extent_mm))
        if extent == 0.0:
            return 1
        return max(2, int(np.ceil(extent / float(target_step_mm))) + 1)

    def _get_image_default_levels(self, name):
        data = self.images[name]['data']
        return self._sanitize_levels((float(np.min(data)), float(np.max(data))))

    def _get_image_center_of_mass_phys(self, image_obj):
        data = np.asarray(image_obj.data, dtype=np.float64)
        if data.ndim < 3:
            return image_obj.transfo_pix2phys([np.zeros(3, dtype=np.float64)])[0]
        if data.ndim > 3:
            data = data[..., 0]

        shape = np.array(data.shape[:3], dtype=np.float64)
        default_center_vox = np.maximum(shape - 1.0, 0.0) / 2.0

        finite_mask = np.isfinite(data)
        if not np.any(finite_mask):
            return image_obj.transfo_pix2phys([default_center_vox])[0]

        finite_values = data[finite_mask]
        weights = np.where(finite_mask, data - np.min(finite_values), 0.0)
        weights = np.maximum(weights, 0.0)
        sum_weights = float(np.sum(weights))

        if sum_weights <= 0.0:
            weights = np.where(finite_mask, np.abs(data), 0.0)
            sum_weights = float(np.sum(weights))

        if sum_weights <= 0.0:
            return image_obj.transfo_pix2phys([default_center_vox])[0]

        x_mass = np.sum(weights, axis=(1, 2))
        y_mass = np.sum(weights, axis=(0, 2))
        z_mass = np.sum(weights, axis=(0, 1))

        com_vox = np.array([
            np.dot(np.arange(weights.shape[0], dtype=np.float64), x_mass) / sum_weights,
            np.dot(np.arange(weights.shape[1], dtype=np.float64), y_mass) / sum_weights,
            np.dot(np.arange(weights.shape[2], dtype=np.float64), z_mass) / sum_weights,
        ], dtype=np.float64)
        return image_obj.transfo_pix2phys([com_vox])[0]

    def align_moving_to_fixed_com(self):
        fixed_img = self._get_fixed_image()
        moving_img = self._get_moving_image()
        if fixed_img is None or moving_img is None:
            return

        fixed_com_phys = self._get_image_center_of_mass_phys(fixed_img)
        moving_com_phys = self._get_image_center_of_mass_phys(moving_img)
        delta_phys = fixed_com_phys - moving_com_phys

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = delta_phys
        parts = [f"{'+' if v >= 0 else ''}{v:.1f}" for v in delta_phys]
        self.push_transform(T, f"Align CoM ({', '.join(parts)}) mm")

    def _moving_center_phys(self):
        moving_img = self._get_moving_image()
        if moving_img is None:
            return None
        moving_shape = np.array(moving_img.data.shape[:3], dtype=np.float64)
        moving_center_vox = np.maximum(moving_shape - 1.0, 0.0) / 2.0
        return moving_img.transfo_pix2phys([moving_center_vox])[0]

    def _write_itk_affine_transform(self, fname_affine, affine_matrix, points_moving_barycenter):
        # ITK text transform expects LPS convention. Our data are handled in RAS, so flip X/Y.
        ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
        affine_itk = ras_to_lps @ np.asarray(affine_matrix, dtype=np.float64) @ ras_to_lps

        rotation_matrix = affine_itk[:3, :3]
        translation_array = affine_itk[:3, 3].reshape(1, 3)
        barycenter = np.asarray(points_moving_barycenter, dtype=np.float64)
        barycenter_itk = np.array([-barycenter[0], -barycenter[1], barycenter[2]], dtype=np.float64)

        text_file = open(fname_affine, 'w')
        text_file.write("#Insight Transform File V1.0\n")
        text_file.write("#Transform 0\n")
        text_file.write("Transform: AffineTransform_double_3_3\n")
        text_file.write("Parameters: %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (
            rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2],
            translation_array[0, 0], translation_array[0, 1], translation_array[0, 2]))
        text_file.write("FixedParameters: %.9f %.9f %.9f\n" % (barycenter_itk[0], barycenter_itk[1], barycenter_itk[2]))
        text_file.close()

    def _parse_itk_affine_transform_text(self, file_text):
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

    def _read_transform_from_file(self, transform_path):
        ext = os.path.splitext(transform_path)[1].lower()

        if ext == '.npy':
            matrix = np.asarray(np.load(transform_path), dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(".npy transform must be a 4x4 matrix.")
            return matrix

        with open(transform_path, 'r', encoding='utf-8') as fobj:
            file_text = fobj.read()

        if "AffineTransform_double_3_3" in file_text:
            matrix = self._parse_itk_affine_transform_text(file_text)
            if matrix is not None:
                return matrix

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

    def load_initial_transform(self, transform_path=None, reset_existing=True):
        if transform_path is None:
            transform_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Initial Transform",
                "",
                "Transform files (*.txt *.tfm *.mat *.npy);;Text files (*.txt *.tfm *.mat);;NumPy files (*.npy);;All Files (*)"
            )

        if not transform_path:
            return

        try:
            matrix = self._read_transform_from_file(transform_path)
            if reset_existing:
                self.reset_stack()
            label = f"Initial ({os.path.basename(transform_path)})"
            self.push_transform(matrix, label)
            self.statusBar().showMessage(f"Initial transform loaded: {transform_path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Load Initial Transform", f"Failed to load transform:\n{exc}")

    def save_current_transform_itk(self):
        moving_img = self._get_moving_image()
        if moving_img is None:
            QMessageBox.warning(self, "Save Transform", "Please load/select a moving image before exporting.")
            return

        suggested = "transform_itk.txt"
        if self.moving_img_name:
            base = os.path.splitext(os.path.basename(self.moving_img_name))[0]
            suggested = f"{base}_to_fixed_itk.txt"

        fname_affine, _ = QFileDialog.getSaveFileName(
            self,
            "Save ITK Transform",
            suggested,
            "ITK Transform (*.txt *.tfm);;All Files (*)"
        )
        if not fname_affine:
            return

        try:
            points_moving_barycenter = self._moving_center_phys()
            self._write_itk_affine_transform(fname_affine, self.affine_matrix, points_moving_barycenter)
            self.statusBar().showMessage(f"Transform saved: {fname_affine}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Save Transform", f"Failed to save transform:\n{exc}")

    def _resample_moving_on_fixed_grid(self, interpolation_mode=1, border='constant'):
        fixed_img = self._get_fixed_image()
        moving_img = self._get_moving_image()
        if fixed_img is None or moving_img is None:
            return None

        nx, ny, nz, _, _, _, _, _ = fixed_img.dim
        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
        indexes_ref = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())), dtype=np.float64)
        fixed_phys = fixed_img.transfo_pix2phys(indexes_ref)

        homogeneous = np.hstack([fixed_phys, np.ones((fixed_phys.shape[0], 1), dtype=np.float64)])
        moving_phys = (np.linalg.inv(self.affine_matrix) @ homogeneous.T).T[:, :3]
        moving_vox = moving_img.transfo_phys2pix(moving_phys, real=False)

        sampled = moving_img.get_values(
            np.array([moving_vox[:, 0], moving_vox[:, 1], moving_vox[:, 2]]),
            interpolation_mode=interpolation_mode,
            border=border
        )

        output = Image(fixed_img)
        if interpolation_mode == 0:
            output.change_type('int32')
        else:
            output.change_type('float32')
        output.data = np.reshape(sampled, (nx, ny, nz))
        return output

    def _ask_interpolation_mode(self):
        options = ["Nearest neighbor", "Linear", "Spline"]
        selected, ok = QInputDialog.getItem(
            self,
            "Interpolation",
            "Choose interpolation mode:",
            options,
            1,
            False,
        )
        if not ok:
            return None
        mapping = {
            "Nearest neighbor": 0,
            "Linear": 1,
            "Spline": 3,
        }
        return mapping[selected]

    def apply_transform_to_moving_and_save(self):
        fixed_img = self._get_fixed_image()
        moving_img = self._get_moving_image()
        if fixed_img is None or moving_img is None:
            QMessageBox.warning(self, "Apply Transform", "Please load/select both fixed and moving images first.")
            return

        interpolation_mode = self._ask_interpolation_mode()
        if interpolation_mode is None:
            return

        if moving_img.absolutepath:
            suggested = add_suffix(moving_img.absolutepath, '_aligned')
        elif self.moving_img_name:
            suggested = f"{self.moving_img_name}_aligned.nii.gz"
        else:
            suggested = "moving_aligned.nii.gz"

        fname_out, _ = QFileDialog.getSaveFileName(
            self,
            "Save Aligned Moving Image",
            suggested,
            "NIfTI (*.nii *.nii.gz);;All Files (*)"
        )
        if not fname_out:
            return

        try:
            aligned_img = self._resample_moving_on_fixed_grid(interpolation_mode=interpolation_mode, border='constant')
            aligned_img.save(fname_out)
            self.statusBar().showMessage(f"Aligned image saved: {fname_out}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Apply Transform", f"Failed to apply/save transform:\n{exc}")

    def _sanitize_levels(self, levels):
        low, high = [float(value) for value in levels]
        if low > high:
            low, high = high, low
        if np.isclose(low, high):
            high = low + 1e-3
        return low, high

    def _normalize_for_display(self, image_2d, levels):
        low, high = self._sanitize_levels(levels)
        scale = high - low
        normalized = (image_2d - low) / scale
        return np.clip(normalized, 0.0, 1.0)

    def _get_view_sampling_spec(self, slice_dim, slice_voxel_pos):
        nx, ny, nz = self.fixed_shape
        spacing_x, spacing_y, spacing_z = self._get_fixed_spacings()

        if slice_dim == 2:
            return {
                'h_dim': 0,
                'v_dim': 1,
                'p_center_vox': np.array([(nx - 1) / 2.0, (ny - 1) / 2.0, slice_voxel_pos], dtype=np.float64),
                'vec_h_vox': np.array([-1.0, 0.0, 0.0], dtype=np.float64),
                'vec_v_vox': np.array([0.0, 1.0, 0.0], dtype=np.float64),
                'plane_size_h': float(nx),
                'plane_size_v': float(ny),
                'spacing_h': float(spacing_x),
                'spacing_v': float(spacing_y),
                'plane_res_h': self._resolution_from_extent_mm((nx - 1) * spacing_x),
                'plane_res_v': self._resolution_from_extent_mm((ny - 1) * spacing_y),
                'flip_crosshair_x': True,
                'flip_crosshair_y': False,
            }
        if slice_dim == 0:
            return {
                'h_dim': 1,
                'v_dim': 2,
                'p_center_vox': np.array([slice_voxel_pos, (ny - 1) / 2.0, (nz - 1) / 2.0], dtype=np.float64),
                'vec_h_vox': np.array([0.0, 1.0, 0.0], dtype=np.float64),
                'vec_v_vox': np.array([0.0, 0.0, 1.0], dtype=np.float64),
                'plane_size_h': float(ny),
                'plane_size_v': float(nz),
                'spacing_h': float(spacing_y),
                'spacing_v': float(spacing_z),
                'plane_res_h': self._resolution_from_extent_mm((ny - 1) * spacing_y),
                'plane_res_v': self._resolution_from_extent_mm((nz - 1) * spacing_z),
                'flip_crosshair_x': False,
                'flip_crosshair_y': False,
            }
        return {
            'h_dim': 0,
            'v_dim': 2,
            'p_center_vox': np.array([(nx - 1) / 2.0, slice_voxel_pos, (nz - 1) / 2.0], dtype=np.float64),
            'vec_h_vox': np.array([1.0, 0.0, 0.0], dtype=np.float64),
            'vec_v_vox': np.array([0.0, 0.0, 1.0], dtype=np.float64),
            'plane_size_h': float(nx),
            'plane_size_v': float(nz),
            'spacing_h': float(spacing_x),
            'spacing_v': float(spacing_z),
            'plane_res_h': self._resolution_from_extent_mm((nx - 1) * spacing_x),
            'plane_res_v': self._resolution_from_extent_mm((nz - 1) * spacing_z),
            'flip_crosshair_x': False,
            'flip_crosshair_y': False,
        }

    def _build_voxel_plane_grid(self, spec):
        half_size_h = (spec['plane_size_h'] - 1.0) / 2.0
        half_size_v = (spec['plane_size_v'] - 1.0) / 2.0
        x_range = np.linspace(-half_size_h, half_size_h, int(spec['plane_res_h']))
        y_range = np.linspace(-half_size_v, half_size_v, int(spec['plane_res_v']))
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        points_vox = (
            spec['p_center_vox']
            + grid_x.ravel()[:, np.newaxis] * spec['vec_h_vox']
            + grid_y.ravel()[:, np.newaxis] * spec['vec_v_vox']
        )
        rect = QRectF(0.0, 0.0, float(spec['plane_size_h']), float(spec['plane_size_v']))
        return points_vox, rect

    def _sample_fixed_slice(self, spec):
        fixed_img = self._get_fixed_image()
        points_vox, rect = self._build_voxel_plane_grid(spec)
        values = fixed_img.get_values(points_vox.T, interpolation_mode=1, transform=False)
        image_2d = values.reshape((int(spec['plane_res_v']), int(spec['plane_res_h'])))
        return image_2d, rect

    def _sample_moving_slice(self, spec):
        moving_img = self._get_moving_image()
        fixed_img = self._get_fixed_image()
        points_vox, rect = self._build_voxel_plane_grid(spec)
        fixed_phys = fixed_img.transfo_pix2phys(points_vox)
        homogeneous = np.hstack([fixed_phys, np.ones((fixed_phys.shape[0], 1), dtype=np.float64)])
        moving_phys = (np.linalg.inv(self.affine_matrix) @ homogeneous.T).T[:, :3]
        moving_vox = moving_img.transfo_phys2pix(moving_phys, real=False)
        values = moving_img.get_values(moving_vox.T, interpolation_mode=1, transform=False)
        image_2d = values.reshape((int(spec['plane_res_v']), int(spec['plane_res_h'])))
        return image_2d, rect

    def _reset_slice_views(self):
        if not self.fixed_img_name:
            return

        center = self._get_fixed_center_vox()
        view_specs = [
            (self.view_axial, 2, center[2]),
            (self.view_sagittal, 0, center[0]),
            (self.view_coronal, 1, center[1]),
        ]

        for widget, slice_dim, slice_pos in view_specs:
            spec = self._get_view_sampling_spec(slice_dim, slice_pos)
            aspect_ratio = spec['spacing_h'] / max(spec['spacing_v'], 1e-12)
            widget.view.setAspectLocked(True, ratio=float(aspect_ratio))
            x_range = (0.0, float(spec['plane_size_h']))
            y_range = (0.0, float(spec['plane_size_v']))
            widget.reset_zoom(x_range, y_range)

    def get_affine_matrix(self):
        result = np.eye(4, dtype=np.float64)
        for entry in self.transform_stack:
            result = entry['matrix'] @ result
        if self._pending_matrix is not None:
            result = self._pending_matrix @ result
        return result

    def update_transform(self):
        self.affine_matrix = self.get_affine_matrix()
        mat_rows = [" ".join(f"{value:7.2f}" for value in row) for row in self.affine_matrix]
        mat_str = "\n".join(mat_rows)
        self.panel.lbl_matrix.setText(f"Affine:\n{mat_str}")
        self.refresh_display()

    def push_transform(self, matrix, label):
        self.transform_stack.append({'matrix': np.array(matrix, dtype=np.float64), 'label': label})
        self.panel.update_transform_stack(self.transform_stack)
        self.update_transform()

    def pop_transform(self):
        if self.transform_stack:
            self.transform_stack.pop()
            self.panel.update_transform_stack(self.transform_stack)
            self.update_transform()

    def reset_stack(self):
        self.transform_stack = []
        self._pending_matrix = None
        self._pending_meta = {}
        self.panel.update_transform_stack(self.transform_stack)
        self.update_transform()

    def set_interaction_mode(self, mode):
        if mode not in {'navigation', 'translation', 'rotation', 'scaling'}:
            return
        self.interaction_mode = mode

        mode_info = {
            'navigation': ('Navigation', '#2d6a4f', None),
            'translation': ('Translation', '#1d4ed8', '#1d4ed8'),
            'rotation': ('Rotation', '#c2410c', '#c2410c'),
            'scaling': ('Scaling', '#0f766e', '#0f766e'),
        }
        label, color, border_color = mode_info[mode]
        self.panel.set_interaction_mode(label, color)

        navigation_enabled = (mode == 'navigation')
        self.view_axial.set_navigation_enabled(navigation_enabled)
        self.view_sagittal.set_navigation_enabled(navigation_enabled)
        self.view_coronal.set_navigation_enabled(navigation_enabled)

        self.view_axial.set_border(border_color)
        self.view_sagittal.set_border(border_color)
        self.view_coronal.set_border(border_color)

        show_crosshair = (mode == 'navigation')
        self.view_axial.set_crosshair_visible(show_crosshair)
        self.view_sagittal.set_crosshair_visible(show_crosshair)
        self.view_coronal.set_crosshair_visible(show_crosshair)

        for key in self._scaling_drag_state:
            self._scaling_drag_state[key] = {'start': None, 'axis': None}

    def _get_image_plane_center(self, widget, slice_dim):
        # Le rect de l'ImageItem est toujours (0, 0, plane_size_h, plane_size_v),
        # donc le centre en coordonnées vue est exactement au milieu.
        spec = self._get_view_sampling_spec(slice_dim, self.cursor_pos[slice_dim])
        return 0.5 * float(spec['plane_size_h']), 0.5 * float(spec['plane_size_v'])

    def _compute_translation_step(self, slice_dim, dx, dy):
        fixed_img = self._get_fixed_image()
        if fixed_img is None:
            return None

        spec = self._get_view_sampling_spec(slice_dim, self.cursor_pos[slice_dim])
        delta_vox = dx * spec['vec_h_vox'] + dy * spec['vec_v_vox']
        delta_phys = fixed_img.transfo_pix2phys(np.asarray([delta_vox], dtype=np.float64), mode='relative')[0]

        acc = self._pending_meta.get('delta_phys', np.zeros(3, dtype=np.float64))
        self._pending_meta['delta_phys'] = acc + delta_phys

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = delta_phys
        return T

    def _compute_rotation_step(self, view_key, widget, slice_dim, x, y, dx, dy):
        cx, cy = self._get_image_plane_center(widget, slice_dim)
        prev = np.array([x - dx - cx, y - dy - cy], dtype=np.float64)
        curr = np.array([x - cx, y - cy], dtype=np.float64)

        if np.linalg.norm(prev) < 1e-6 or np.linalg.norm(curr) < 1e-6:
            return None

        cross_z = prev[0] * curr[1] - prev[1] * curr[0]
        dot = prev[0] * curr[0] + prev[1] * curr[1]
        angle_deg = np.degrees(np.arctan2(cross_z, dot))

        if view_key == 'sagittal':
            angle_deg = -angle_deg

        self._pending_meta['angle_deg'] = self._pending_meta.get('angle_deg', 0.0) + angle_deg
        self._pending_meta['view_key'] = view_key

        angle_rad = np.radians(angle_deg)
        directions = self._get_fixed_directions()
        axis_idx = {'axial': 2, 'sagittal': 0, 'coronal': 1}[view_key]
        ax, ay, az = directions[axis_idx]

        c, s = np.cos(angle_rad), np.sin(angle_rad)
        t = 1.0 - c
        R3 = np.array([
            [t*ax*ax + c,     t*ax*ay - s*az, t*ax*az + s*ay],
            [t*ax*ay + s*az,  t*ay*ay + c,    t*ay*az - s*ax],
            [t*ax*az - s*ay,  t*ay*az + s*ax, t*az*az + c   ],
        ], dtype=np.float64)

        center_phys = self._get_fixed_center_phys()
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R3
        M[:3, 3] = center_phys - R3 @ center_phys
        return M

    def _compute_scaling_step(self, view_key, slice_dim, x, y, dx, dy, is_start, is_finish):
        state = self._scaling_drag_state[view_key]

        if state['axis'] is None:
            total = np.array([x, y], dtype=np.float64) - state['start']
            if np.linalg.norm(total) >= 0.5:
                state['axis'] = 'h' if abs(total[0]) >= abs(total[1]) else 'v'

        if state['axis'] is None:
            if is_finish:
                self._scaling_drag_state[view_key] = {'start': None, 'axis': None}
            return None

        spec = self._get_view_sampling_spec(slice_dim, self.cursor_pos[slice_dim])
        axis_dim = spec['h_dim'] if state['axis'] == 'h' else spec['v_dim']
        axis_delta = dx if state['axis'] == 'h' else dy

        factor = max(0.01, float(np.exp(0.01 * axis_delta)))
        self._pending_meta['scale_factor'] = self._pending_meta.get('scale_factor', 1.0) * factor
        self._pending_meta['scale_axis'] = axis_dim

        center_phys = self._get_fixed_center_phys()
        S = np.eye(4, dtype=np.float64)
        S[axis_dim, axis_dim] = factor
        S[:3, 3] = center_phys - S[:3, :3] @ center_phys

        if is_finish:
            self._scaling_drag_state[view_key] = {'start': None, 'axis': None}
        return S

    def _build_transform_label(self):
        mode = self.interaction_mode
        meta = self._pending_meta
        if mode == 'translation':
            d = meta.get('delta_phys', np.zeros(3))
            parts = [f"{'+' if v >= 0 else ''}{v:.1f}" for v in d]
            return f"T ({', '.join(parts)}) mm"
        elif mode == 'rotation':
            angle = meta.get('angle_deg', 0.0)
            view = meta.get('view_key', '?')
            sign = '+' if angle >= 0 else ''
            return f"R {sign}{angle:.1f}° ({view})"
        elif mode == 'scaling':
            factor = meta.get('scale_factor', 1.0)
            axis = meta.get('scale_axis', 0)
            return f"S {'XYZ'[axis]} ×{factor:.3f}"
        return "Transform"

    def handle_view_drag(self, view_key, widget, slice_dim, x, y, dx, dy, is_start, is_finish):
        if self.interaction_mode == 'navigation':
            return
        if self._get_moving_image() is None:
            return

        if is_start:
            self._pending_matrix = np.eye(4, dtype=np.float64)
            self._pending_meta = {}
            if self.interaction_mode == 'scaling':
                self._scaling_drag_state[view_key] = {
                    'start': np.array([x, y], dtype=np.float64),
                    'axis': None,
                }

        T_step = None
        if self.interaction_mode == 'translation':
            T_step = self._compute_translation_step(slice_dim, dx, dy)
        elif self.interaction_mode == 'rotation':
            T_step = self._compute_rotation_step(view_key, widget, slice_dim, x, y, dx, dy)
        elif self.interaction_mode == 'scaling':
            T_step = self._compute_scaling_step(view_key, slice_dim, x, y, dx, dy, is_start, is_finish)

        if T_step is not None and self._pending_matrix is not None:
            self._pending_matrix = T_step @ self._pending_matrix
            self.update_transform()

        if is_finish:
            if self._pending_matrix is not None and not np.allclose(self._pending_matrix, np.eye(4)):
                label = self._build_transform_label()
                self.transform_stack.append({'matrix': self._pending_matrix, 'label': label})
                self.panel.update_transform_stack(self.transform_stack)
            self._pending_matrix = None
            self._pending_meta = {}
            self.update_transform()

    def refresh_display(self):
        fixed_img = self._get_fixed_image()
        if fixed_img is None:
            return

        fixed_data = self.images[self.fixed_img_name]['data']
        moving_img = self._get_moving_image()

        fixed_box_corners_phys = self._get_image_box_corners_phys(fixed_img)
        fixed_box_segments_phys = self._phys_points_to_segments(fixed_box_corners_phys)

        moving_box_segments_phys = None
        if moving_img is not None:
            moving_corners_phys = self._get_image_box_corners_phys(moving_img)
            moving_corners_in_fixed_phys = self._transform_phys_points(moving_corners_phys, self.affine_matrix)
            moving_box_segments_phys = self._phys_points_to_segments(moving_corners_in_fixed_phys)

        x, y, z = self.cursor_pos

        marker_visible = (self.interaction_mode == 'rotation')

        def update_view(widget, slice_dim, slice_val, h_dim, v_dim):
            spec = self._get_view_sampling_spec(slice_dim, slice_val)
            aspect_ratio = spec['spacing_h'] / max(spec['spacing_v'], 1e-12)
            widget.view.setAspectLocked(True, ratio=float(aspect_ratio))
            fixed_slice, rect = self._sample_fixed_slice(spec)
            fixed_slice_norm = self._normalize_for_display(fixed_slice, self.fixed_levels)

            fixed_box_segments_2d = self._project_phys_segments_to_view(fixed_box_segments_phys, spec)
            moving_box_segments_2d = self._project_phys_segments_to_view(moving_box_segments_phys, spec)
            widget.set_projected_boxes(fixed_box_segments_2d, moving_box_segments_2d)

            cross_x = self.fixed_shape[h_dim] - 1 - self.cursor_pos[h_dim] if spec.get('flip_crosshair_x', False) else self.cursor_pos[h_dim]
            cross_y = self.fixed_shape[v_dim] - 1 - self.cursor_pos[v_dim] if spec.get('flip_crosshair_y', False) else self.cursor_pos[v_dim]
            widget.set_crosshair(cross_x, cross_y)

            center_x, center_y = self._get_image_plane_center(widget, slice_dim)
            widget.set_rotation_center_marker(center_x, center_y, marker_visible)

            # Affichage Fixe
            if not hasattr(widget, 'fixed_item'):
                item = pg.ImageItem()
                widget.view.addItem(item);
                item.setZValue(0)
                widget.fixed_item = item

            widget.fixed_item.setImage(fixed_slice_norm, autoLevels=False)
            widget.fixed_item.setRect(rect)
            widget.fixed_item.setLevels([0.0, 1.0])
            widget.fixed_item.setLookupTable(get_lut_for_colormap('gray'))
            widget.fixed_item.setOpacity(self.fixed_opacity)
            widget.fixed_item.setVisible(self.fixed_visible)

            # Affichage Mobile (Overlay)
            if moving_img is not None:
                if not hasattr(widget, 'moving_item'):
                    item = pg.ImageItem()
                    widget.view.addItem(item);
                    item.setZValue(1)
                    item.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                    widget.moving_item = item

                resampled, _ = self._sample_moving_slice(spec)
                moving_slice_norm = self._normalize_for_display(resampled, self.moving_levels)
                widget.moving_item.setImage(moving_slice_norm, autoLevels=False)
                widget.moving_item.setRect(rect)
                widget.moving_item.setLookupTable(get_lut_for_colormap('gray'))
                widget.moving_item.setLevels([0.0, 1.0])
                widget.moving_item.setOpacity(self.moving_opacity)
                widget.moving_item.setVisible(self.moving_visible)
            elif hasattr(widget, 'moving_item'):
                widget.moving_item.setVisible(False)

        update_view(self.view_axial, 2, z, 0, 1)
        update_view(self.view_sagittal, 0, x, 1, 2)
        update_view(self.view_coronal, 1, y, 0, 2)

        # Mise à jour 3D
        point_phys = fixed_img.transfo_pix2phys([np.array([x, y, z], dtype=np.float64)])[0]
        directions = self._get_fixed_directions()
        plane_size = 0.5 * np.max(self._get_fixed_extent_mm())
        self.view_3d.update_cursor_3d(point_phys)
        # Affiche les plans centrés sur le curseur
        self.view_3d.update_trajectory_planes(
            point_phys,
            directions[2],
            directions[0],
            directions[1],
            size=max(plane_size, 1.0),
        )
        self.view_3d.set_image_boxes(fixed_box_segments_phys, moving_box_segments_phys)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Z and (modifiers & Qt.KeyboardModifier.ControlModifier):
            self.pop_transform()
            event.accept()
            return

        if key == Qt.Key.Key_T:
            self.set_interaction_mode('translation')
            event.accept()
            return
        if key == Qt.Key.Key_R:
            self.set_interaction_mode('rotation')
            event.accept()
            return
        if key == Qt.Key.Key_S:
            self.set_interaction_mode('scaling')
            event.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.set_interaction_mode('navigation')
            event.accept()
            return

        if self._get_moving_image() is None:
            super().keyPressEvent(event)
            return

        step = 5.0 if (modifiers & Qt.KeyboardModifier.ShiftModifier) else 1.0
        directions = self._get_fixed_directions()
        delta = None

        if key == Qt.Key.Key_Left:
            delta = -step * directions[0]
        elif key == Qt.Key.Key_Right:
            delta = step * directions[0]
        elif key == Qt.Key.Key_Up:
            delta = step * directions[1]
        elif key == Qt.Key.Key_Down:
            delta = -step * directions[1]
        elif key == Qt.Key.Key_PageUp:
            delta = step * directions[2]
        elif key == Qt.Key.Key_PageDown:
            delta = -step * directions[2]

        if delta is not None:
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = delta
            parts = [f"{'+' if v >= 0 else ''}{v:.1f}" for v in delta]
            self.push_transform(T, f"T ({', '.join(parts)}) mm")
            event.accept()
            return

        super().keyPressEvent(event)