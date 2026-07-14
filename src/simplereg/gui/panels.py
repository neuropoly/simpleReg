from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLabel,
                             QComboBox, QGroupBox, QSlider, QDoubleSpinBox,
                             QPushButton, QScrollArea, QSizePolicy, QListWidget,
                             QListWidgetItem, QHBoxLayout, QCheckBox)
from PyQt6.QtCore import Qt

from .utils import STANDARD_CMAPS, QUALITATIVE_CMAPS


class RegistrationControlPanel(QWidget):
    """
    Panneau de contrôle pour gérer les paramètres de transformation affine
    (Translation, Rotation, Scaling) et la sélection des images.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        outer_layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        outer_layout.addWidget(self.scroll_area)

        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(8)
        self.scroll_area.setWidget(self.content_widget)

        # --- 1. Sélection des images ---
        grp_sel, l_sel = self.create_collapsible_group("Selection")
        self.combo_fixed = QComboBox()
        self.combo_moving = QComboBox()
        self.combo_fixed.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_moving.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_fixed.setMinimumContentsLength(12)
        self.combo_moving.setMinimumContentsLength(12)
        self.btn_load_fixed = QPushButton("Load Fixed")
        self.btn_load_moving = QPushButton("Load Moving")
        self.btn_toggle_fixed = QPushButton("Hide Fixed")
        self.btn_toggle_fixed.setCheckable(True)
        self.btn_toggle_fixed.setChecked(True)
        self.btn_toggle_moving = QPushButton("Hide Moving")
        self.btn_toggle_moving.setCheckable(True)
        self.btn_toggle_moving.setChecked(True)
        for btn in (self.btn_load_fixed, self.btn_load_moving, self.btn_toggle_fixed, self.btn_toggle_moving):
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        l_sel.addWidget(QLabel("Fixed (Ref):"), 0, 0)
        l_sel.addWidget(self.combo_fixed, 0, 1)
        l_sel.addWidget(QLabel("Moving (Src):"), 1, 0)
        l_sel.addWidget(self.combo_moving, 1, 1)
        l_sel.addWidget(self.btn_load_fixed, 2, 0)
        l_sel.addWidget(self.btn_load_moving, 2, 1)
        l_sel.addWidget(self.btn_toggle_fixed, 3, 0)
        l_sel.addWidget(self.btn_toggle_moving, 3, 1)
        l_sel.setColumnStretch(0, 1)
        l_sel.setColumnStretch(1, 1)
        self.layout.addWidget(grp_sel)

        # --- 2. Paramètres de transformation ---
        # --- 2. Pile de transformations ---
        self._create_transform_stack_group()

        self.lbl_interaction_mode = QLabel("Mode: Navigation")
        self.lbl_interaction_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_interaction_mode.setStyleSheet(
            "font-weight: 700; font-size: 13px; "
            "color: #ffffff; background-color: #2d6a4f; "
            "border: 1px solid #1b4332; border-radius: 4px; padding: 6px;"
        )
        self.layout.addWidget(self.lbl_interaction_mode)

        self.display_spinboxes = {}
        self.create_display_group()
        self.create_opacity_group()
        self.create_edge_group()
        self.create_color_group()

        # --- 3. Bouton Reset & Matrice ---
        self.btn_align_com = QPushButton("Align Moving\nto Fixed (COM)")
        self.btn_align_com.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.layout.addWidget(self.btn_align_com)

        self.btn_load_initial_transform = QPushButton("Load Initial Transform")
        self.btn_load_initial_transform.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.layout.addWidget(self.btn_load_initial_transform)

        self.btn_save_transform = QPushButton("Save Transform\n(ITK .txt)")
        self.btn_save_transform.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.layout.addWidget(self.btn_save_transform)

        self.btn_apply_and_save = QPushButton("Apply Transform to Moving\n& Save")
        self.btn_apply_and_save.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.layout.addWidget(self.btn_apply_and_save)

        self.lbl_matrix = QLabel("Matrix:\nIdentity")
        self.lbl_matrix.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_matrix.setStyleSheet("font-family: monospace; font-size: 10px; border: 1px solid #555; padding: 5px;")
        self.layout.addWidget(self.lbl_matrix)

        self.btn_shortcuts_help = QPushButton("Keyboard Help")
        self.btn_shortcuts_help.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.layout.addWidget(self.btn_shortcuts_help)

        self.layout.addStretch()

    def create_collapsible_group(self, title):
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)

        btn_toggle = QPushButton("Hide")
        btn_toggle.setCheckable(True)
        btn_toggle.setChecked(True)
        btn_toggle.setMaximumWidth(72)
        btn_toggle.toggled.connect(lambda checked, b=btn_toggle: b.setText("Hide" if checked else "Show"))

        content_widget = QWidget()
        content_layout = QGridLayout(content_widget)

        btn_toggle.toggled.connect(content_widget.setVisible)

        group_layout.addWidget(btn_toggle, alignment=Qt.AlignmentFlag.AlignRight)
        group_layout.addWidget(content_widget)
        return group, content_layout

    def set_fixed_visibility_label(self, visible):
        self.btn_toggle_fixed.setText("Hide Fixed" if visible else "Show Fixed")

    def set_moving_visibility_label(self, visible):
        self.btn_toggle_moving.setText("Hide Moving" if visible else "Show Moving")

    def create_display_group(self):
        group, layout = self.create_collapsible_group("Display Levels")

        field_specs = [
            ("fixed_min", "Fixed Min"),
            ("fixed_max", "Fixed Max"),
            ("moving_min", "Moving Min"),
            ("moving_max", "Moving Max"),
        ]

        for row, (key, label) in enumerate(field_specs):
            layout.addWidget(QLabel(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-1e9, 1e9)
            spin.setSingleStep(1.0)
            spin.setMaximumWidth(90)
            spin.valueChanged.connect(lambda _value, k=key: self.on_display_level_changed(k))
            self.display_spinboxes[key] = spin
            layout.addWidget(spin, row, 1)
        layout.setColumnStretch(0, 1)
        self.layout.addWidget(group)

    def on_display_level_changed(self, key):
        window = self.window()
        if hasattr(window, 'update_display_levels'):
            window.update_display_levels(key)

    def set_display_level_controls(self, prefix, min_value, max_value):
        min_spin = self.display_spinboxes[f"{prefix}_min"]
        max_spin = self.display_spinboxes[f"{prefix}_max"]

        lower_bound = min(min_value, max_value)
        upper_bound = max(min_value, max_value)
        span = max(1.0, upper_bound - lower_bound)
        margin = max(1.0, 0.1 * span)

        for spin, value in ((min_spin, min_value), (max_spin, max_value)):
            spin.blockSignals(True)
            spin.setRange(lower_bound - margin, upper_bound + margin)
            spin.setValue(value)
            spin.blockSignals(False)

    def get_display_levels(self, prefix):
        return (
            self.display_spinboxes[f"{prefix}_min"].value(),
            self.display_spinboxes[f"{prefix}_max"].value(),
        )

    def create_opacity_group(self):
        group, layout = self.create_collapsible_group("Opacity")

        self.opacity_fixed_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_fixed_slider.setRange(0, 100)
        self.opacity_fixed_slider.setValue(100)
        self.opacity_fixed_label = QLabel("100%")
        self.opacity_fixed_label.setFixedWidth(36)

        self.opacity_moving_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_moving_slider.setRange(0, 100)
        self.opacity_moving_slider.setValue(60)
        self.opacity_moving_label = QLabel("60%")
        self.opacity_moving_label.setFixedWidth(36)

        layout.addWidget(QLabel("Fixed:"), 0, 0)
        layout.addWidget(self.opacity_fixed_slider, 0, 1)
        layout.addWidget(self.opacity_fixed_label, 0, 2)
        layout.addWidget(QLabel("Moving:"), 1, 0)
        layout.addWidget(self.opacity_moving_slider, 1, 1)
        layout.addWidget(self.opacity_moving_label, 1, 2)

        self.opacity_fixed_slider.valueChanged.connect(lambda v: self.on_opacity_changed('fixed', v))
        self.opacity_moving_slider.valueChanged.connect(lambda v: self.on_opacity_changed('moving', v))
        self.layout.addWidget(group)

    def create_color_group(self):
        group, layout = self.create_collapsible_group("Color Map")

        self.colormap_fixed_combo = QComboBox()
        self.colormap_moving_combo = QComboBox()
        self.colormap_fixed_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.colormap_moving_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.colormap_fixed_combo.setMinimumContentsLength(12)
        self.colormap_moving_combo.setMinimumContentsLength(12)

        self.colormap_options = list(STANDARD_CMAPS) + list(QUALITATIVE_CMAPS)
        for combo in (self.colormap_fixed_combo, self.colormap_moving_combo):
            combo.addItems(self.colormap_options)

        layout.addWidget(QLabel("Fixed:"), 0, 0)
        layout.addWidget(self.colormap_fixed_combo, 0, 1)
        layout.addWidget(QLabel("Moving:"), 1, 0)
        layout.addWidget(self.colormap_moving_combo, 1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        self.colormap_fixed_combo.currentTextChanged.connect(lambda name: self.on_colormap_changed('fixed', name))
        self.colormap_moving_combo.currentTextChanged.connect(lambda name: self.on_colormap_changed('moving', name))
        self.layout.addWidget(group)

    def on_colormap_changed(self, which, name):
        window = self.window()
        if hasattr(window, 'update_colormap'):
            window.update_colormap(which, name)

    def set_colormap_controls(self, prefix, name):
        combo = self.colormap_fixed_combo if prefix == 'fixed' else self.colormap_moving_combo
        index = combo.findText(name)
        if index < 0:
            index = combo.findText('gray')
        combo.blockSignals(True)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def get_colormap(self, prefix):
        combo = self.colormap_fixed_combo if prefix == 'fixed' else self.colormap_moving_combo
        return combo.currentText()

    def create_edge_group(self):
        group, layout = self.create_collapsible_group("Edge Enhancement")

        self.edge_enhance_checkbox = QCheckBox("Enhance moving edges")
        self.edge_enhance_checkbox.setChecked(False)

        self.edge_enhance_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_enhance_slider.setRange(0, 200)
        self.edge_enhance_slider.setValue(100)
        self.edge_enhance_label = QLabel("1.00x")
        self.edge_enhance_label.setFixedWidth(45)

        layout.addWidget(self.edge_enhance_checkbox, 0, 0, 1, 3)
        layout.addWidget(QLabel("Strength:"), 1, 0)
        layout.addWidget(self.edge_enhance_slider, 1, 1)
        layout.addWidget(self.edge_enhance_label, 1, 2)

        self.edge_enhance_checkbox.toggled.connect(self.on_edge_enhancement_changed)
        self.edge_enhance_slider.valueChanged.connect(self.on_edge_strength_changed)
        self.layout.addWidget(group)

    def on_edge_strength_changed(self, value):
        strength = value / 100.0
        self.edge_enhance_label.setText(f"{strength:.2f}x")
        self.on_edge_enhancement_changed()

    def on_edge_enhancement_changed(self):
        window = self.window()
        if hasattr(window, 'update_edge_enhancement'):
            window.update_edge_enhancement(
                self.edge_enhance_checkbox.isChecked(),
                self.edge_enhance_slider.value() / 100.0,
            )

    def set_edge_enhancement_controls(self, enabled, strength):
        slider_value = int(round(float(strength) * 100.0))
        slider_value = max(0, min(200, slider_value))
        self.edge_enhance_checkbox.blockSignals(True)
        self.edge_enhance_slider.blockSignals(True)
        self.edge_enhance_checkbox.setChecked(bool(enabled))
        self.edge_enhance_slider.setValue(slider_value)
        self.edge_enhance_checkbox.blockSignals(False)
        self.edge_enhance_slider.blockSignals(False)
        self.edge_enhance_label.setText(f"{slider_value / 100.0:.2f}x")

    def on_opacity_changed(self, which, value):
        label = self.opacity_fixed_label if which == 'fixed' else self.opacity_moving_label
        label.setText(f"{value}%")
        window = self.window()
        if hasattr(window, 'update_opacity'):
            window.update_opacity(which, value / 100.0)

    def _create_transform_stack_group(self):
        group = QGroupBox("Transform Stack")
        grp_layout = QVBoxLayout(group)
        grp_layout.setContentsMargins(6, 6, 6, 6)
        grp_layout.setSpacing(4)

        self.transform_list = QListWidget()
        self.transform_list.setMaximumHeight(160)
        self.transform_list.setStyleSheet(
            "QListWidget { font-family: monospace; font-size: 11px; "
            "background-color: #1a1a1a; border: 1px solid #444; }"
            "QListWidget::item { padding: 3px; }"
            "QListWidget::item:last-child { background-color: #2a3a2a; }"
        )
        grp_layout.addWidget(self.transform_list)

        btn_row = QHBoxLayout()
        self.btn_remove_last = QPushButton("↩ Remove last  (Ctrl+Z)")
        self.btn_remove_last.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_remove_last.clicked.connect(self._on_remove_last)
        btn_row.addWidget(self.btn_remove_last)

        self.btn_reset_stack = QPushButton("✕ Reset all")
        self.btn_reset_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_reset_stack.clicked.connect(self._on_reset_stack)
        btn_row.addWidget(self.btn_reset_stack)

        grp_layout.addLayout(btn_row)
        self.layout.addWidget(group)

    def update_transform_stack(self, stack):
        self.transform_list.clear()
        for i, entry in enumerate(stack):
            item = QListWidgetItem(f"{i + 1:2d}.  {entry['label']}")
            self.transform_list.addItem(item)
        self.transform_list.scrollToBottom()

    def _on_remove_last(self):
        window = self.window()
        if hasattr(window, 'pop_transform'):
            window.pop_transform()

    def _on_reset_stack(self):
        window = self.window()
        if hasattr(window, 'reset_stack'):
            window.reset_stack()

    def set_interaction_mode(self, mode_name, accent_color):
        self.lbl_interaction_mode.setText(f"Mode: {mode_name}")
        self.lbl_interaction_mode.setStyleSheet(
            "font-weight: 700; font-size: 13px; "
            "color: #ffffff; "
            f"background-color: {accent_color}; "
            "border: 1px solid #1f1f1f; border-radius: 4px; padding: 6px;"
        )