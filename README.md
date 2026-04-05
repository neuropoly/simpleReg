# SimpleReg

SimpleReg est une application de recalage manuel 3D (rigide/affine) pour images médicales NIfTI, avec visualisation 2D/3D et export des transformations.

## Dépendances

- Python >= 3.11
- PyQt6
- pyqtgraph
- numpy
- scipy
- nibabel
- scikit-image
- transforms3d
- matplotlib

## Installation

1. Cloner le dépôt puis se placer à la racine du projet.
   ```bash
   git clone https://github.com/neuropoly/simplereg.git
   cd simplereg
   ```

2. Créer et activer un environnement Python.
   - venv
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   - conda
   ```bash
   conda create -n simplereg python=3.11
   conda activate simplereg
   ```

3. Installer en mode développement.
   ```bash
   python3 -m pip install -e .
   ```

## Lancement

Lancer l'application :

```bash
python scripts/start_app.py
```

Lancer avec une transformation initiale (`.txt/.tfm/.mat/.npy`) :

```bash
python scripts/start_app.py --initial-transform /path/to/transform.txt
```

Par défaut, la transformation initiale réinitialise la pile de transformations.
Pour l'ajouter à la pile existante :

```bash
python scripts/start_app.py --initial-transform /path/to/transform.txt --append-initial-transform
```

## Structure du répertoire

```text
simpleReg/
├── README.md
├── scripts/
│   └── start_app.py
└── src/
    └── simplereg/
        ├── core/
        │   └── image.py
        └── gui/
            ├── panels.py
            ├── utils.py
            ├── viewers.py
            └── window.py
```

## Rôle des principaux modules

- `scripts/start_app.py` : point d'entrée de l'application (arguments CLI, thème Qt, ouverture de la fenêtre principale).
- `src/simplereg/gui/window.py` : logique de recalage (gestion des images, pile de transformations, interactions clavier/souris, export).
- `src/simplereg/gui/viewers.py` : widgets de visualisation 2D (axial/sagittal/coronal) et vue 3D.
- `src/simplereg/gui/panels.py` : panneau de contrôle (sélection d'images, niveaux, opacité, pile de transformations, actions d'export).
- `src/simplereg/gui/utils.py` : utilitaires de colormaps/LUT pour l'affichage.
- `src/simplereg/core/image.py` : abstraction image (chargement NIfTI, orientation, conversions voxel/physique, interpolation).

## Fonctionnalités

- Chargement d'images NIfTI (`.nii`, `.nii.gz`) en fixe (référence) et mobile (source).
- Réorientation des volumes en espace commun pour une manipulation cohérente.
- Affichage multiplanaire (axial, sagittal, coronal) avec curseur synchronisé.
- Superposition fixe/mobile avec contrôle d'opacité et réglage indépendant des niveaux d'intensité.
- Visualisation 3D avec curseur, plans de coupe et boîtes englobantes des volumes fixe/mobile.
- Alignement automatique initial du mobile sur le fixe via centre de masse (CoM).
- Manipulation manuelle du mobile en translation, rotation et mise à l'échelle.
- Pile de transformations (historique), annulation du dernier mouvement et réinitialisation complète.
- Import d'une transformation initiale (formats ITK texte, matrices numériques texte, `.npy` 4x4).
- Export de la transformation courante au format ITK (`.txt/.tfm`) avec conversion RAS/LPS.
- Application de la transformation sur la grille du fixe et sauvegarde de l'image alignée.
- Choix de l'interpolation pour le resampling (nearest, linear, spline).

## Raccourcis utiles

- `T` : mode translation
- `R` : mode rotation
- `S` : mode scaling
- `Esc` : retour mode navigation
- `Ctrl+Z` : retirer la dernière transformation
- Flèches / `PageUp` / `PageDown` : translations discrètes (maintenir `Shift` pour un pas plus grand)


