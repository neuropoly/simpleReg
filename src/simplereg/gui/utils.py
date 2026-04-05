import numpy as np
import colorsys
import pyqtgraph as pg
import matplotlib.colors as mcolors

ATLAS_LABELS = {
    0: "Background", 2: "Left cerebral white matter", 3: "Left cerebral cortex", 4: "Left lateral ventricle",
    5: "Left inferior lateral ventricle", 7: "Left cerebellum white matter", 8: "Left cerebellum cortex",
    10: "Left thalamus", 11: "Left caudate", 12: "Left putamen", 13: "Left pallidum", 14: "3rd ventricle",
    15: "4th ventricle", 16: "Brain-stem", 17: "Left hippocampus", 18: "Left amygdala", 24: "CSF",
    26: "Left accumbens area", 28: "Left ventral DC", 41: "Right cerebral white matter", 42: "Right cerebral cortex",
    43: "Right lateral ventricle", 44: "Right inferior lateral ventricle", 46: "Right cerebellum white matter",
    47: "Right cerebellum cortex", 49: "Right thalamus", 50: "Right caudate", 51: "Right putamen",
    52: "Right pallidum", 53: "Right hippocampus", 54: "Right amygdala", 58: "Right accumbens area",
    60: "Right ventral DC"
}

STANDARD_CMAPS = ['gray', 'red_vessels', 'viridis', 'plasma', 'inferno', 'magma', 'hot', 'jet']
QUALITATIVE_CMAPS = ['atlas_glasbey', 'atlas_vivid', 'atlas_tab20', 'atlas_random']


def get_lut_for_colormap(name, data_max=255):
    if name == 'gray': return None
    if name == 'red_vessels':
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[:, 0] = 255;
        lut[:, 3] = np.linspace(0, 255, 256)
        return lut
    if name.startswith('atlas'):
        n_colors = int(data_max) + 1
        lut = np.zeros((n_colors, 4), dtype=np.uint8)
        if name == 'atlas_glasbey':
            glasbey_colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
                (255, 128, 0), (128, 0, 255), (0, 128, 0), (128, 0, 0), (0, 0, 128), (128, 128, 0),
                (0, 128, 128), (128, 0, 128), (192, 192, 192), (255, 192, 203), (210, 180, 140),
                (255, 215, 0), (0, 255, 127), (75, 0, 130), (255, 165, 0), (240, 230, 140),
                (127, 255, 212), (64, 224, 208), (0, 191, 255), (30, 144, 255), (138, 43, 226),
                (255, 20, 147), (139, 69, 19), (250, 128, 114)
            ]
            for i in range(1, n_colors):
                c = glasbey_colors[(i - 1) % len(glasbey_colors)]
                lut[i] = (*c, 200)
        elif name == 'atlas_vivid':
            np.random.seed(42)
            for i in range(1, n_colors):
                h, s, v = np.random.rand(), np.random.uniform(0.8, 1.0), np.random.uniform(0.8, 1.0)
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                lut[i] = (int(r * 255), int(g * 255), int(b * 255), 200)
        elif name == 'atlas_tab20':
            base_colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), (44, 160, 44),
                           (152, 223, 138), (214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213),
                           (140, 86, 75), (196, 156, 148), (227, 119, 194), (247, 182, 210), (127, 127, 127),
                           (199, 199, 199), (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
            for i in range(1, n_colors):
                c = base_colors[(i - 1) % len(base_colors)]
                lut[i] = (*c, 150)
        else:
            np.random.seed(42);
            lut[1:] = np.random.randint(50, 255, size=(n_colors - 1, 4));
            lut[1:, 3] = 255
        lut[0] = [0, 0, 0, 0]
        return lut
    try:
        cmap = pg.colormap.get(name)
        # alpha=True force une LUT RGBA; sans cela on obtient souvent RGB uniquement
        # et l'acces lut[:, 3] echoue silencieusement dans le except.
        lut = cmap.getLookupTable(0.0, 1.0, 256, alpha=True)
        lut[0, 3] = 0  # Pixels au niveau minimum → transparents (fond invisible)
        return lut
    except:
        return None


def get_mpl_colormap_from_name(name, data_max=255):
    """Crée une colormap Matplotlib synchronisée avec celle de PyQtGraph."""
    if name == 'gray':
        return 'gray'

    # On récupère la même LUT que pour les vues 2D
    lut = get_lut_for_colormap(name, data_max)

    if lut is None:
        return 'viridis'  # Fallback

    # Conversion PyQtGraph (0-255) vers Matplotlib (0.0-1.0)
    lut_normalized = lut.astype(float) / 255.0
    # Création de l'objet Colormap
    return mcolors.ListedColormap(lut_normalized)
