import logging
from logging import config

import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table, perimeter
from skimage.filters import threshold_otsu, threshold_multiotsu

from Deeplearning.util.functions import figure_preproduction, find_source_from_sample_id

WAVELENGTH = 632.8e-9
CELL_MAPS = ("rbc", "wbc", "platelet")


def area_extract(array, coordinates, otsu_bins=20, expand_distance=5):
    h_upper, w_upper = array.shape
    labeled_area = np.zeros_like(array)
    # Extracting the available area from the give boxes
    for lbl, (x, y, w, h) in enumerate(coordinates):
        # Decide weather all the area inside the box is available
        xmin, xmax = np.clip(x - w / 2, 0, None).astype(int), np.clip(x + w / 2, 0, w_upper).astype(int)
        ymin, ymax = np.clip(y - h / 2, 0, None).astype(int), np.clip(y + h / 2, 0, h_upper).astype(int)

        # Extract available areas from boxes
        strict_area = array[ymin:ymax, xmin:xmax]
        otsu_thres = threshold_otsu(strict_area, nbins=otsu_bins)

        # Labeling available areas
        y_index, x_index = (strict_area >= otsu_thres).nonzero()
        labeled_area[y_index + ymin, x_index + xmin] = lbl + 1

    # Expanding the extracted areas
    expanded_labels = expand_labels(labeled_area, distance=expand_distance)
    # Re-thresholding of the expanded areas using Otsu-threshold
    for lbl in range(1, np.max(expanded_labels).astype(int)):
        lbl_area = (array - np.min(array)) * (expanded_labels == lbl)
        regions = np.digitize(lbl_area, bins=threshold_multiotsu(lbl_area))
        expanded_labels[regions == 1] = -lbl
        expanded_labels[regions == 2] = lbl
    return expanded_labels


def cross_minimal_euclidean_distance(coordinates):
    """ Select outlier cells through Computing Euclidean distance.
    Outputs
    -------
    minimals: list
        [(original_cell_id, pointed_cell_id, distance)]
    """
    x, y = np.transpose(coordinates)
    xx1, xx2 = np.meshgrid(x, x)
    yy1, yy2 = np.meshgrid(y, y)
    distances = np.sqrt((xx1 - xx2) ** 2 + (yy1 - yy2) ** 2)
    distances += np.eye(len(distances)) * np.max(distances)  # Maximal the distance with itself

    minimal_distances = np.min(distances, axis=-1)
    return np.arange(len(coordinates)), minimal_distances


def outlier_cells_visual(img, outlier_cells, outlier_minimals, figsize=(7, 7)):
    fig, ax = figure_preproduction(figsize=figsize)
    ax.imshow(img, cmap="gray")
    colors = ("green", "blue", "purple")
    for cell, mini in zip(outlier_cells, outlier_minimals):
        (x, y, w, h, lbl), (_io, _ip, dist) = cell, mini
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=False, color=colors[lbl]))
        ax.annotate(f"{dist.item():.1f}", xy=(x - 5, y + 3), xycoords="data", size=8, color="red")
    plt.show()


def select_outlier_cells(coordinates, labels, distance_thres=22, specified_cell_type="rbc", cell_maps=CELL_MAPS):
    _, mini_distances =   cross_minimal_euclidean_distance(coordinates[:, :2])
    outlier_indexes,  =   np.where(mini_distances > distance_thres)
    correct_celltype  =   labels[outlier_indexes] == cell_maps.index(specified_cell_type)
    return outlier_indexes[correct_celltype]


def extract_cell_region_properties(phase, coordinates, indexes=None, otsu_bins=20, prop_names=None) -> dict:
    """ Extract recognized RBC cells with the dominant areas and
    produce specified properties into and pandas dict for every phase image.
    Returns
    -------
    properties_tables: list
        [properties_table_0: {'area', 'bbox', ...}, properties_table_1, ...]
    """

    def volume(image, intensity): return np.sum(intensity * image)

    def mch(image): return 10 * WAVELENGTH * np.sum(image > 0) / (2 * np.pi * 0.002)

    def mean_phase_shift(image, intensity): return np.sum(intensity) / np.sum(image)

    def form_factor(image): return 4 * np.pi * np.sum(image) / np.square(perimeter(image))

    # Extracting area labels
    if indexes is not None:
        coordinates = coordinates[indexes]

    area_labels = area_extract(phase, coordinates, otsu_bins).astype(int)

    if not prop_names:
        prop_names = ("area", "bbox", "eccentricity", "perimeter", "intensity_image")

    return regionprops_table(area_labels * (area_labels > 0),
                             intensity_image=phase,
                             properties=prop_names,
                             extra_properties=(form_factor, volume, mean_phase_shift, mch))


def outlier_rbc_analysis(phase_float_array, coordinates, labels):
    # Read the real Phase map of each sample recognized by previous programs
    coordinates, labels = [elm.numpy() for elm in (coordinates, labels)]
    # Select outlier cells and only focus on rbc
    outlier_rbc_indexes = select_outlier_cells(coordinates, labels, specified_cell_type="rbc")
    # Extract rbc cell region properties based on cell coordinates and phase
    rbc_properties = extract_cell_region_properties(phase_float_array, coordinates, indexes=outlier_rbc_indexes)
    return rbc_properties


if __name__ == "__main__":
    # Logging Config ---------------------------------------------------
    logging.config.fileConfig(".\\log\\config\\evaluate.conf")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    pass
