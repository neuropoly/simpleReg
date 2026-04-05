from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPen, QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QApplication
from skimage import measure
import numpy as np


# Fallback minimal pour permettre le lancement même sans module core.models.
class Layer:
    TYPE_LABEL = 'label'
    TYPE_VESSEL = 'vessel'

from .utils import get_lut_for_colormap


class CustomViewBox(pg.ViewBox):
    """ViewBox personnalisée : Clic Droit = Pan, Clic Gauche = ignoré (géré par SliceWidget)"""

    sig_left_drag = pyqtSignal(float, float, float, float, bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMenuEnabled(False)

    def mouseDragEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            diff = self.mapToView(ev.pos()) - self.mapToView(ev.lastPos())
            self.translateBy(x=-diff.x(), y=-diff.y())
        elif ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            current = self.mapToView(ev.pos())
            last = self.mapToView(ev.lastPos())
            self.sig_left_drag.emit(
                float(current.x()),
                float(current.y()),
                float(current.x() - last.x()),
                float(current.y() - last.y()),
                bool(ev.isStart()),
                bool(ev.isFinish()),
            )
        else:
            super().mouseDragEvent(ev)


class SliceWidget(pg.GraphicsLayoutWidget):
    sig_clicked = pyqtSignal(float, float)
    sig_drag = pyqtSignal(float, float, float, float, bool, bool)

    def __init__(self, title, color='y'):
        super().__init__()
        self.title_label = self.addLabel(title, row=0, col=0, color=color)

        self.view = CustomViewBox()
        self.addItem(self.view, row=1, col=0)
        self.view.setAspectLocked(True)
        self.view.setMouseEnabled(x=True, y=True)

        # Image composite pour le mode trajectoire
        self.traj_view_item = pg.ImageItem()
        self.traj_view_item.setZValue(-100)
        self.traj_view_item.setVisible(False)
        self.view.addItem(self.traj_view_item)

        # Lignes du curseur (Crosshair)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=color)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=color)
        self.v_line.setZValue(20000)
        self.h_line.setZValue(20000)
        self.view.addItem(self.v_line)
        self.view.addItem(self.h_line)

        self.rotation_center_item = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen(255, 255, 0, 240, width=2),
            brush=pg.mkBrush(255, 255, 0, 120),
            symbol='x'
        )
        self.rotation_center_item.setZValue(20003)
        self.rotation_center_item.setVisible(False)
        self.view.addItem(self.rotation_center_item)

        # Points pour le mode Ortho
        self.entry_item = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 200), symbol='o')
        self.target_item = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200), symbol='o')
        self.entry_item.setZValue(20002)
        self.target_item.setZValue(20002)
        self.view.addItem(self.entry_item)
        self.view.addItem(self.target_item)

        self.other_traj_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 100),
                                                  symbol='x')
        self.other_traj_item.setZValue(20001)
        self.view.addItem(self.other_traj_item)

        # --- Éléments graphiques pour le Tube 2D ---
        pen = QPen(QColor(0, 150, 255, 200), 2)  # Bleu clair, épaisseur 2
        pen.setCosmetic(True)  # Épaisseur constante

        # 1. Cercle (vue transverse)
        self.tube_outline_item = pg.QtWidgets.QGraphicsEllipseItem(0, 0, 1, 1)
        self.tube_outline_item.setPen(pen)
        self.tube_outline_item.setVisible(False)
        self.tube_outline_item.setZValue(10000)
        self.view.addItem(self.tube_outline_item)

        # 2. Lignes (vues longitudinales)
        self.tube_line_1 = pg.QtWidgets.QGraphicsLineItem()
        self.tube_line_1.setPen(pen)
        self.tube_line_1.setVisible(False)
        self.tube_line_1.setZValue(10000)
        self.view.addItem(self.tube_line_1)

        self.tube_line_2 = pg.QtWidgets.QGraphicsLineItem()
        self.tube_line_2.setPen(pen)
        self.tube_line_2.setVisible(False)
        self.tube_line_2.setZValue(10000)
        self.view.addItem(self.tube_line_2)

        # Wireframes de position des volumes (fixe et mobile) projetes dans la vue 2D.
        self.fixed_box_item = pg.PlotDataItem(pen=pg.mkPen(255, 255, 0, 220, width=2), connect='finite')
        self.moving_box_item = pg.PlotDataItem(pen=pg.mkPen(255, 165, 0, 220, width=2), connect='finite')
        self.fixed_box_item.setZValue(15000)
        self.moving_box_item.setZValue(15001)
        self.view.addItem(self.fixed_box_item)
        self.view.addItem(self.moving_box_item)

        self.view.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.view.scene().sigMouseMoved.connect(self.mouse_moved)
        self.view.sig_left_drag.connect(self.sig_drag.emit)

        self.current_slice_val = 0
        self.fade_distance = 10.0
        self.navigation_enabled = True

        # On crée 4 TextItem pour Haut, Bas, Gauche, Droite
        # anchor permet de centrer le texte par rapport au point d'ancrage
        self.lbl_top = pg.TextItem("", color=color, anchor=(0.5, 0))
        self.lbl_bottom = pg.TextItem("", color=color, anchor=(0.5, 1))
        self.lbl_left = pg.TextItem("", color=color, anchor=(0, 0.5))
        self.lbl_right = pg.TextItem("", color=color, anchor=(1, 0.5))

        for lbl in [self.lbl_top, self.lbl_bottom, self.lbl_left, self.lbl_right]:
            self.view.addItem(lbl, ignoreBounds=True)
            lbl.setVisible(False)  # Cachés par défaut jusqu'à configuration
            lbl.setZValue(20005)  # Au-dessus de tout

            # Connecter le changement de zoom/pan pour garder les labels aux bords
        self.view.sigRangeChanged.connect(self.update_labels_pos)

    def set_border(self, color_str=None):
        if color_str:
            self.setStyleSheet(f"SliceWidget {{ border: 3px solid {color_str}; }}")
        else:
            self.setStyleSheet("")

    def set_title(self, text, color='w'):
        self.title_label.setText(text, color=color)

    def set_navigation_enabled(self, active):
        self.navigation_enabled = active

    def set_current_slice(self, val):
        self.current_slice_val = val

    def _get_visual_style(self, point_3d, axis_v_idx, color_rgb):
        if point_3d is None: return None
        dist = abs(point_3d[axis_v_idx] - self.current_slice_val)
        if dist > self.fade_distance: return None
        factor = 1.0 - (dist / self.fade_distance)
        size = 10 + (10 * factor)
        alpha = int(50 + (205 * factor))
        return {
            'size': size,
            'brush': pg.mkBrush(*color_rgb, alpha),
            'pen': pg.mkPen(*color_rgb, alpha)
        }

    def reset_zoom(self, x_range, y_range):
        """
        Réinitialise la vue sur une plage donnée.
        x_range et y_range sont des tuples (min, max).
        """
        # setRange applique le zoom. padding=0 assure qu'on colle exactement aux bords demandés.
        self.view.setRange(xRange=x_range, yRange=y_range, padding=0.0)

    def set_traj_points(self, active_traj, h_axis, v_axis, ortho_axis, flip_x_dim=None):
        """
        Affiche les points d'entrée et de cible de la trajectoire active.
        flip_x_dim : Si fourni (int), inverse la coordonnée X (Display = Dim - 1 - X).
        """
        pt_e = active_traj.entry_pos if active_traj else None
        style_e = self._get_visual_style(pt_e, ortho_axis, (0, 255, 0))
        if style_e:
            ex, ey = pt_e[h_axis], pt_e[v_axis]
            # --- CORRECTION : Inversion ---
            if flip_x_dim is not None:
                ex = flip_x_dim - 1 - ex
            # ------------------------------
            self.entry_item.setData(pos=[[ex, ey]],
                                    size=style_e['size'], brush=style_e['brush'], pen=style_e['pen'])
            self.entry_item.setVisible(True)
        else:
            self.entry_item.setVisible(False)

        pt_t = active_traj.target_pos if active_traj else None
        style_t = self._get_visual_style(pt_t, ortho_axis, (255, 0, 0))
        if style_t:
            tx, ty = pt_t[h_axis], pt_t[v_axis]
            # --- CORRECTION : Inversion ---
            if flip_x_dim is not None:
                tx = flip_x_dim - 1 - tx
            # ------------------------------
            self.target_item.setData(pos=[[tx, ty]],
                                     size=style_t['size'], brush=style_t['brush'], pen=style_t['pen'])
            self.target_item.setVisible(True)
        else:
            self.target_item.setVisible(False)

    def update_trajectories_display(self, active_traj, all_trajectories, h_axis, v_axis, ortho_axis, flip_x_dim=None):
        """
        Affiche les points de toutes les autres trajectoires (fantômes).
        flip_x_dim : Si fourni (int), inverse la coordonnée X.
        """
        spots = []
        for t in all_trajectories:
            if t is active_traj: continue

            # Entrée
            style_e = self._get_visual_style(t.entry_pos, ortho_axis, t.color)
            if style_e:
                ex, ey = t.entry_pos[h_axis], t.entry_pos[v_axis]
                if flip_x_dim is not None: ex = flip_x_dim - 1 - ex
                spots.append({'pos': [ex, ey], 'size': style_e['size'] * 0.7,
                              'brush': style_e['brush'], 'symbol': 'o'})

            # Cible
            style_t = self._get_visual_style(t.target_pos, ortho_axis, t.color)
            if style_t:
                tx, ty = t.target_pos[h_axis], t.target_pos[v_axis]
                if flip_x_dim is not None: tx = flip_x_dim - 1 - tx
                spots.append({'pos': [tx, ty], 'size': style_t['size'] * 0.7,
                              'brush': style_t['brush'], 'symbol': 'o'})

        if spots:
            self.other_traj_item.setData(spots)
            self.other_traj_item.setVisible(True)
        else:
            self.other_traj_item.setVisible(False)

    def set_crosshair(self, x, y):
        self.v_line.setPos(x)
        self.h_line.setPos(y)

    def set_crosshair_visible(self, visible):
        self.v_line.setVisible(bool(visible))
        self.h_line.setVisible(bool(visible))

    def set_rotation_center_marker(self, x, y, visible):
        self.rotation_center_item.setVisible(bool(visible))
        if visible:
            self.rotation_center_item.setData(pos=[[float(x), float(y)]])

    def set_tube_display(self, visible, center_mm, radius_mm):
        """Affiche la coupe TRANSVERSE (cercle) du tube."""
        self.tube_outline_item.setVisible(visible)
        if visible:
            x = center_mm[0] - radius_mm
            y = center_mm[1] - radius_mm
            diameter = 2 * radius_mm
            self.tube_outline_item.setRect(x, y, diameter, diameter)
        else:
            self.tube_outline_item.setRect(0, 0, 0, 0)

    def set_longitudinal_tube_display(self, visible, radius_mm, bounds_mm, orientation='horizontal'):
        """
        Affiche la coupe LONGITUDINALE (lignes) du tube.
        'bounds_mm' est un tuple (min, max) indiquant le début et la fin du tube
        par rapport au centre de l'image.
        """
        self.tube_line_1.setVisible(visible)
        self.tube_line_2.setVisible(visible)

        if visible and bounds_mm is not None:
            start, stop = bounds_mm

            if orientation == 'horizontal':
                # Le tube court le long de l'axe X (entre x=start et x=stop)
                # Les parois sont à y = +radius et y = -radius
                self.tube_line_1.setLine(start, radius_mm, stop, radius_mm)
                self.tube_line_2.setLine(start, -radius_mm, stop, -radius_mm)

            elif orientation == 'vertical':
                # Le tube court le long de l'axe Y (entre y=start et y=stop)
                # Les parois sont à x = -radius et x = +radius
                self.tube_line_1.setLine(-radius_mm, start, -radius_mm, stop)
                self.tube_line_2.setLine(radius_mm, start, radius_mm, stop)
        else:
            self.tube_line_1.setLine(0, 0, 0, 0)
            self.tube_line_2.setLine(0, 0, 0, 0)

    def set_projected_boxes(self, fixed_segments_2d, moving_segments_2d=None):
        def _segments_to_xy(segments):
            if segments is None or len(segments) == 0:
                return np.array([]), np.array([])

            xs = []
            ys = []
            for seg in segments:
                p0, p1 = seg
                xs.extend([float(p0[0]), float(p1[0]), np.nan])
                ys.extend([float(p0[1]), float(p1[1]), np.nan])
            return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

        x_fixed, y_fixed = _segments_to_xy(fixed_segments_2d)
        self.fixed_box_item.setData(x=x_fixed, y=y_fixed)
        self.fixed_box_item.setVisible(len(x_fixed) > 0)

        x_moving, y_moving = _segments_to_xy(moving_segments_2d)
        self.moving_box_item.setData(x=x_moving, y=y_moving)
        self.moving_box_item.setVisible(len(x_moving) > 0)

    def mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.navigation_enabled:
            if self.view.sceneBoundingRect().contains(event.scenePos()):
                mouse_point = self.view.mapSceneToView(event.scenePos())
                self.sig_clicked.emit(mouse_point.x(), mouse_point.y())

    def mouse_moved(self, pos):
        if self.navigation_enabled and (QApplication.mouseButtons() & Qt.MouseButton.LeftButton):
            if self.view.sceneBoundingRect().contains(pos):
                mouse_point = self.view.mapSceneToView(pos)
                self.sig_clicked.emit(mouse_point.x(), mouse_point.y())

    def setup_orientation_labels(self, top, bottom, left, right):
        """Définit les lettres pour les 4 directions."""
        self.lbl_top.setText(top)
        self.lbl_bottom.setText(bottom)
        self.lbl_left.setText(left)
        self.lbl_right.setText(right)

        # On les rend visibles
        self.set_orientation_visible(True)
        self.update_labels_pos()

    def set_orientation_visible(self, visible):
        """Affiche ou cache les labels (ex: caché en mode trajectoire)."""
        self.lbl_top.setVisible(visible)
        self.lbl_bottom.setVisible(visible)
        self.lbl_left.setVisible(visible)
        self.lbl_right.setVisible(visible)

    def update_labels_pos(self):
        """Recalcule la position des labels pour qu'ils collent aux bords de la vue visible."""
        if not self.lbl_top.isVisible(): return

        # Récupérer les limites actuelles de la vue (ce qui est affiché à l'écran)
        # viewRange retourne [[xmin, xmax], [ymin, ymax]]
        (xmin, xmax), (ymin, ymax) = self.view.viewRange()

        # Calculer le centre et les dimensions
        width = xmax - xmin
        height = ymax - ymin
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        # Marge de 2% pour ne pas coller exactement au bord
        margin_x = width * 0.02
        margin_y = height * 0.02

        # Positionner les labels
        self.lbl_top.setPos(x_center, ymax - margin_y)
        self.lbl_bottom.setPos(x_center, ymin + margin_y)
        self.lbl_left.setPos(xmin + margin_x, y_center)
        self.lbl_right.setPos(xmax - margin_x, y_center)

class BrainViewer3D(gl.GLViewWidget):
    sig_point_moved = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.setCameraPosition(distance=250, elevation=30, azimuth=45)
        self.setBackgroundColor('k')

        # La lumière ambiante éclaire les faces même si elles ne sont pas face à la source
        # Valeur par défaut souvent basse (ex: 0.2). On la monte à 0.4 ou 0.5.
        self.opts['ambient'] = (0.6, 0.6, 0.6, 1.0)

        # La lumière directionnelle (specular/diffuse).
        # On peut augmenter son intensité (valeurs > 1.0 possibles mais attention à la saturation)
        # self.opts['light'] = (1.0, 1.0, 1.0, 1.0)
        # light_pos = (100, 100, 100)  # Coordonnées (x, y, z) de la lumière
        # self.opts['lightPosition'] = light_pos
        # self.light = gl.GLScatterPlotItem(pos=np.array([light_pos]), color=(1, 1, 0, 1), size=15, pxMode=True)
        # self.addItem(self.light)

        self.cursor_item = gl.GLAxisItem()
        self.cursor_item.setSize(20, 20, 20)
        self.addItem(self.cursor_item)

        self.center_of_rotation = np.array([0, 0, 0])
        self.traj_items = {}

        self.last_mouse_pos = None
        self.interaction_mode = 'none'

        self.entry_mesh = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(0, 1, 0, 1), size=15, pxMode=True)
        self.target_mesh = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=15, pxMode=True)
        self.addItem(self.entry_mesh)
        self.addItem(self.target_mesh)

        dummy_verts = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)
        dummy_faces = np.array([[0, 1, 2]], dtype=int)

        self.plane_transverse = gl.GLMeshItem(vertexes=dummy_verts, faces=dummy_faces, shader='shaded',
                                              color=(0, 1, 0, 0.2), glOptions='translucent')
        self.plane_transverse_border = gl.GLLinePlotItem(color=(0, 1, 0, 1), width=3)

        self.plane_ortho_1 = gl.GLMeshItem(vertexes=dummy_verts, faces=dummy_faces, shader='shaded',
                                           color=(1, 0, 0, 0.2), glOptions='translucent')
        self.plane_ortho_1_border = gl.GLLinePlotItem(color=(1, 0, 0, 1), width=3)

        self.plane_ortho_2 = gl.GLMeshItem(vertexes=dummy_verts, faces=dummy_faces, shader='shaded',
                                           color=(0, 0, 1, 0.2), glOptions='translucent')
        self.plane_ortho_2_border = gl.GLLinePlotItem(color=(0, 0, 1, 1), width=3)

        self.fixed_box_3d = gl.GLLinePlotItem(color=(1.0, 1.0, 0.0, 1.0), width=2, mode='lines')
        self.moving_box_3d = gl.GLLinePlotItem(color=(1.0, 0.6, 0.0, 1.0), width=2, mode='lines')
        self.addItem(self.fixed_box_3d)
        self.addItem(self.moving_box_3d)
        self.fixed_box_3d.setVisible(False)
        self.moving_box_3d.setVisible(False)

        for item in [self.plane_transverse, self.plane_transverse_border,
                     self.plane_ortho_1, self.plane_ortho_1_border,
                     self.plane_ortho_2, self.plane_ortho_2_border]:
            self.addItem(item)
            item.setVisible(False)

    def set_interaction_mode(self, mode):
        self.interaction_mode = mode

    def mousePressEvent(self, ev):
        self.last_mouse_pos = ev.pos()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        modifiers = QApplication.keyboardModifiers()
        if (self.interaction_mode in ['entry', 'target'] and
                (ev.buttons() & Qt.MouseButton.LeftButton) and
                (modifiers & Qt.KeyboardModifier.ShiftModifier)):

            diff = ev.pos() - self.last_mouse_pos
            self.last_mouse_pos = ev.pos()
            cam_dist = self.cameraParams()['distance']
            factor = cam_dist * 0.001
            dx = diff.x() * factor
            dy = -diff.y() * factor
            self.sig_point_moved.emit((dx, dy), self.interaction_mode)
        else:
            super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0: return
        scale_factor = 1.05 if delta < 0 else 0.95
        current_dist = self.cameraParams()['distance']
        new_dist = max(1.0, current_dist * scale_factor)
        self.setCameraPosition(distance=new_dist)
        ev.accept()

    def set_mesh(self, layer, volume):
        try:
            if np.max(volume) == 0: return

            if layer.img_type == Layer.TYPE_LABEL:
                level = 0.5
                color = (0.2, 0.8, 0.2, 0.6)
                layer.base_3d_color = color
                shader = 'shaded'
                options = 'translucent'
            elif layer.img_type == Layer.TYPE_VESSEL:
                level = 0.3
                color = (1.0, 0.0, 0.0, 1.0)
                layer.base_3d_color = color
                shader = 'shaded'
                options = 'translucent'
            else:
                valid_voxels = volume[volume > 0]
                level = np.mean(valid_voxels) if len(valid_voxels) > 0 else np.mean(volume)
                color = (0.5, 0.5, 0.5, 1.0)
                layer.base_3d_color = color
                shader = 'shaded'  # 'balloon', 'viewNormalColor', 'normalColor', 'shaded', 'edgeHilight', 'heightColor', 'pointSprite'
                options = 'translucent'  # 'additive', 'opaque', 'translucent'

            # Gestion multi-label
            if layer.img_type == Layer.TYPE_LABEL:
                lut = get_lut_for_colormap(layer.colormap, np.max(volume))
                unique_labels = np.unique(volume)
                mesh_list = []

                for lbl in unique_labels:
                    if lbl == 0: continue
                    mask = (volume == lbl).astype(float)
                    if np.sum(mask) < 10: continue
                    try:
                        verts, faces, normals, values = measure.marching_cubes(mask, 0.5)
                        verts = verts - self.center_of_rotation
                        c_byte = lut[int(lbl)]
                        color_rgba = (c_byte[0] / 255.0, c_byte[1] / 255.0, c_byte[2] / 255.0, layer.opacity)

                        mesh = gl.GLMeshItem(vertexes=verts, faces=faces, smooth=True,
                                             shader=shader, color=color_rgba, glOptions=options)
                        mesh.base_color = color_rgba
                        self.addItem(mesh)
                        mesh_list.append(mesh)
                    except:
                        continue
                layer.mesh_item = mesh_list

            else:
                verts, faces, normals, values = measure.marching_cubes(volume, level)
                verts = verts - self.center_of_rotation

                mesh = gl.GLMeshItem(vertexes=verts, faces=faces, smooth=True,
                                     shader=shader, color=color, glOptions=options)

                if not hasattr(layer, 'base_3d_color'):
                    layer.base_3d_color = color

                mesh.base_color = layer.base_3d_color
                if layer.mesh_item: self.removeItem(layer.mesh_item)
                layer.mesh_item = mesh
                self.addItem(mesh)

        except Exception as e:
            print(f"Err mesh: {e}")

    def update_layer_visuals(self, layer):
        if not layer.mesh_item: return

        items = layer.mesh_item if isinstance(layer.mesh_item, list) else [layer.mesh_item]

        for item in items:
            item.setVisible(layer.visible_3d)
            if layer.visible_3d and hasattr(item, 'base_color'):
                r, g, b, base_alpha = item.base_color
                factor = 1.0
                if layer.img_type == Layer.TYPE_STRUCTURAL: factor = 0.3
                new_alpha = max(0.01, base_alpha * layer.opacity * factor)
                item.setColor((r, g, b, new_alpha))

    def update_all_trajectories(self, trajectories, active_traj):
        current_keys = list(self.traj_items.keys())
        for t in current_keys:
            if t not in trajectories:
                self.removeItem(self.traj_items[t])
                del self.traj_items[t]

        for t in trajectories:
            if t not in self.traj_items:
                item = gl.GLLinePlotItem(width=3)
                self.addItem(item)
                self.traj_items[t] = item

            item = self.traj_items[t]
            if t.entry_pos is not None and t.target_pos is not None:
                pts = np.array([t.entry_pos, t.target_pos]) - self.center_of_rotation
                item.setData(pos=pts)
                item.setVisible(True)
                if t == active_traj:
                    item.setData(color=(1, 1, 0, 1), width=5)
                else:
                    c = t.color
                    item.setData(color=(c[0] / 255, c[1] / 255, c[2] / 255, 0.5), width=2)
            else:
                item.setVisible(False)

        if active_traj:
            if active_traj.entry_pos is not None:
                pos = active_traj.entry_pos - self.center_of_rotation
                self.entry_mesh.setData(pos=np.array([pos]))
                self.entry_mesh.setVisible(True)
            else:
                self.entry_mesh.setVisible(False)

            if active_traj.target_pos is not None:
                pos = active_traj.target_pos - self.center_of_rotation
                self.target_mesh.setData(pos=np.array([pos]))
                self.target_mesh.setVisible(True)
            else:
                self.target_mesh.setVisible(False)
        else:
            self.entry_mesh.setVisible(False)
            self.target_mesh.setVisible(False)

    def update_cursor_3d(self, point_3d):
        cx, cy, cz = self.center_of_rotation
        x, y, z = np.asarray(point_3d, dtype=float)
        self.cursor_item.resetTransform()
        self.cursor_item.translate(x - cx, y - cy, z - cz)

    def update_trajectory_planes(self, p_center, vec_T, vec_U, vec_V, size=50):
        if p_center is None: return
        p_center_gl = p_center - self.center_of_rotation

        p1 = p_center_gl - size * vec_U - size * vec_V
        p2 = p_center_gl + size * vec_U - size * vec_V
        p3 = p_center_gl + size * vec_U + size * vec_V
        p4 = p_center_gl - size * vec_U + size * vec_V
        verts_trans = np.array([p1, p2, p3, p4])
        faces = np.array([[0, 1, 2], [0, 2, 3]])

        # --- CORRECTION : Utilisation de setMeshData avec vertexes et faces ---
        self.plane_transverse.setMeshData(vertexes=verts_trans, faces=faces)
        self.plane_transverse_border.setData(pos=np.array([p1, p2, p3, p4, p1]), width=3, color=(0, 1, 0, 1))

        p1 = p_center_gl - size * vec_T - size * vec_U
        p2 = p_center_gl + size * vec_T - size * vec_U
        p3 = p_center_gl + size * vec_T + size * vec_U
        p4 = p_center_gl - size * vec_T + size * vec_U
        verts_ortho1 = np.array([p1, p2, p3, p4])

        self.plane_ortho_1.setMeshData(vertexes=verts_ortho1, faces=faces)
        self.plane_ortho_1_border.setData(pos=np.array([p1, p2, p3, p4, p1]), width=3, color=(1, 0, 0, 1))

        p1 = p_center_gl - size * vec_T - size * vec_V
        p2 = p_center_gl + size * vec_T - size * vec_V
        p3 = p_center_gl + size * vec_T + size * vec_V
        p4 = p_center_gl - size * vec_T + size * vec_V
        verts_ortho2 = np.array([p1, p2, p3, p4])

        self.plane_ortho_2.setMeshData(vertexes=verts_ortho2, faces=faces)
        self.plane_ortho_2_border.setData(pos=np.array([p1, p2, p3, p4, p1]), width=3, color=(0, 0, 1, 1))

    def show_trajectory_planes(self, show):
        for item in [self.plane_transverse, self.plane_transverse_border,
                     self.plane_ortho_1, self.plane_ortho_1_border,
                     self.plane_ortho_2, self.plane_ortho_2_border]:
            item.setVisible(show)

    def set_image_boxes(self, fixed_segments_phys, moving_segments_phys=None):
        def _segments_to_pos(segments):
            if segments is None or len(segments) == 0:
                return None
            pts = np.asarray(segments, dtype=np.float64).reshape(-1, 3)
            return pts - self.center_of_rotation

        fixed_pos = _segments_to_pos(fixed_segments_phys)
        if fixed_pos is not None and len(fixed_pos) > 0:
            self.fixed_box_3d.setData(pos=fixed_pos, color=(1.0, 1.0, 0.0, 1.0), width=2, mode='lines')
            self.fixed_box_3d.setVisible(True)
        else:
            self.fixed_box_3d.setVisible(False)

        moving_pos = _segments_to_pos(moving_segments_phys)
        if moving_pos is not None and len(moving_pos) > 0:
            self.moving_box_3d.setData(pos=moving_pos, color=(1.0, 0.6, 0.0, 1.0), width=2, mode='lines')
            self.moving_box_3d.setVisible(True)
        else:
            self.moving_box_3d.setVisible(False)