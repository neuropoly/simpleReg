import sys
import os
import argparse

# Configuration du chemin d'accès (PYTHONPATH)
# On récupère le dossier courant du script (scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# On remonte d'un niveau pour avoir la racine (simpleReg/)
root_dir = os.path.dirname(current_dir)
# On ajoute src/simplereg au path pour que les imports "from gui..." et "from core..." fonctionnent
package_root = os.path.join(root_dir, "src", "simplereg")
sys.path.append(package_root)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from gui.window import RegistrationApp


DEFAULT_FIXED_IMAGE = "/Users/benjamindeleener/code/BrainThrough/brainthrough/data/sub-2381129/2381129\\,_Assouline\\,_Jacob_AX_T1_3D_TFE_MEDTRONIC_C+_20251027142339_901.nii"
DEFAULT_MOVING_IMAGE = "/Users/benjamindeleener/code/BrainThrough/brainthrough/data/sub-2381129/2381129\\,_Assouline\\,_Jacob_NEURONAVROSA_HSJ_20251027160354_2.nii"
DEFAULT_INITIAL_TRANSFORM = None


def resolve_startup_image_path(path):
    if os.path.exists(path):
        return path

    unescaped_path = path.replace("\\,", ",")
    if os.path.exists(unescaped_path):
        return unescaped_path

    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SimpleReg application")
    parser.add_argument(
        "--initial-transform",
        dest="initial_transform",
        default=DEFAULT_INITIAL_TRANSFORM,
        help="Path to an initial transform file (.txt/.tfm/.mat/.npy)",
    )
    parser.add_argument(
        "--append-initial-transform",
        action="store_true",
        help="Append initial transform to current stack instead of resetting it",
    )
    args, qt_args = parser.parse_known_args()

    app = QApplication([sys.argv[0], *qt_args])
    pg.setConfigOptions(imageAxisOrder='row-major')

    # Application du thème sombre (Fusion Style)
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

    # Lancement
    window = RegistrationApp()

    fixed_path = resolve_startup_image_path(DEFAULT_FIXED_IMAGE)
    moving_path = resolve_startup_image_path(DEFAULT_MOVING_IMAGE)

    fixed_name = window.register_image(fixed_path)
    if fixed_name:
        window.panel.combo_fixed.setCurrentText(fixed_name)

    moving_name = window.register_image(moving_path)
    if moving_name:
        window.panel.combo_moving.setCurrentText(moving_name)

    if args.initial_transform:
        initial_transform_path = resolve_startup_image_path(args.initial_transform)
        window.load_initial_transform(
            transform_path=initial_transform_path,
            reset_existing=not args.append_initial_transform,
        )

    window.show()
    sys.exit(app.exec())