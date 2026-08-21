import sys
import os
import argparse

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
import pyqtgraph as pg

try:
    from simplereg.gui.window import RegistrationApp
except ModuleNotFoundError:
    # Fallback pour execution locale sans installation pip.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    package_src = os.path.join(root_dir, "src")
    if package_src not in sys.path:
        sys.path.append(package_src)
    from simplereg.gui.window import RegistrationApp

PATH_DATA = "/Volumes/BDL/sub-3354641_Anne_Brodeur/meld/merged"
DEFAULT_FIXED_IMAGE = os.path.join(PATH_DATA, "sub-3354641preop_T1w.nii.gz")
DEFAULT_MOVING_IMAGE = os.path.join(PATH_DATA, "sub-3354641postop20250825_T1w.nii")

# suffix = "_normto14m.nii.gz

DEFAULT_INITIAL_TRANSFORM = None


def resolve_startup_image_path(path):
    if os.path.exists(path):
        return path

    unescaped_path = path.replace("\\,", ",")
    if os.path.exists(unescaped_path):
        return unescaped_path

    return path


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

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
    args, qt_args = parser.parse_known_args(argv)

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
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())