import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table, perimeter
from skimage.filters import threshold_otsu, threshold_multiotsu

from .util.functions import figure_preproduction

WAVELENGTH = 632.8e-9
CELL_MAPS = ("rbc", "wbc", "platelet")


def cross_minimal_eudistance(coordinates):
    """ Select outlier cells through Computing Euclidean distance.
    Outputs
    -------
    minimals: list
        [(original_cell_id, pointed_cell_id, distance)] """
    x, y = np.transpose(coordinates)
    xx1, xx2 = np.meshgrid(x, x)
    yy1, yy2 = np.meshgrid(y, y)
    distances = np.sqrt((xx1 - xx2) ** 2 + (yy1 - yy2) ** 2)
    distances += np.eye(len(distances)) * np.max(distances)  # Maximal the distance with itself
    minimal_distances = np.min(distances, axis=-1)
    return np.arange(len(coordinates)), minimal_distances


def outlier_cells_visual(img, outlier_cells, outlier_minimals=None, figsize=(7, 7)):
    colors = ("lime", "blue", "purple")
    fig, ax = figure_preproduction(figsize=figsize)
    ax.imshow(img, cmap="gray")
    for i, cell in enumerate(outlier_cells):
        x, y, w, h, lbl = cell
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=2, fill=False, color=colors[int(lbl)]))
        if outlier_minimals is not None:
            _io, _ip, dist = outlier_minimals[i]
            ax.annotate(f"{dist.item():.1f}", xy=(x - 5, y + 3), xycoords="data", size=8, color="red")
    plt.show()


def area_extract(array, coordinates, otsu_bins=20, expand_distance=5):
    h_upper, w_upper = array.shape
    labeled_area = np.zeros_like(array)
    # Extracting the available area from the given boxes
    for lbl, (x, y, w, h, _) in enumerate(coordinates):
        # Decide weather all the area inside the box is available
        # Notice: the 'lbl' used here is different from the 'label' used in object annotating
        # during the object recognition processing, but only for the pixel labeling.
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


def select_outlier_cells(boxes, distance_thres=22, specified_type_idx=0):
    """
    Parameters
    ---------
    boxes: numpy.ndarray
        [N, x-y-w-h-lbl]
    distance_thres: int
    specified_type_idx: int
        represents the selected cell type, 0 for rbc, 1 for wbc, 2 for platelet.
    """
    _, mini_distances = cross_minimal_eudistance(boxes[:, :2])
    outlier_indexes, = np.where(mini_distances > distance_thres)
    celltype_correct = boxes[:, 4][outlier_indexes] == specified_type_idx
    return outlier_indexes[celltype_correct]


def remove_cells_closed_to_edge(boxes, indexes, image_size=(320, 320)):
    """Those cells too close to the edge of image fov is not suitable
    for cell analysis, thus remove then."""
    removes = []
    for idx in indexes:
        x, y, w, h, lbl = boxes[idx]
        large_size = max([w, h])
        x_included = 0 <= x - large_size / 2 and x + large_size / 2 < image_size[1]
        y_included = 0 <= y - large_size / 2 and y + large_size / 2 < image_size[0]
        if not x_included or not y_included:
            # part of cell outside the border
            removes.append(idx)
    indexes = indexes.tolist()
    for rm in removes:
        indexes.remove(rm)
    return np.array(indexes)


def cell_region_properties(phase, boxes, indexes=None, otsu_bins=20, properties=None) -> dict:
    """ Extract recognized RBC cells with the dominant areas and
    produce specified properties into and pandas dict for every phase image.
    Returns
    -------
    properties_tables: list
        [properties_table_0: {'area', 'bbox', ...}, properties_table_1, ...] """
    def volume(image, intensity): return np.sum(intensity * image)

    def mch(image): return 10 * WAVELENGTH * np.sum(image > 0) / (2 * np.pi * 0.002)

    def mean_phase_shift(image, intensity): return np.sum(intensity) / np.sum(image)

    def form_factor(image): return 4 * np.pi * np.sum(image) / np.square(perimeter(image))

    # Extracting area labels
    if indexes is not None:
        boxes = boxes[indexes]
    area_labels = area_extract(phase, boxes, otsu_bins).astype(int)
    area_labels = area_labels * (area_labels > 0)
    if not properties:
        properties = ("image", "label", "area", "bbox", "eccentricity", "perimeter", "intensity_image")
    prop_dict = regionprops_table(area_labels,
                                  intensity_image=phase,
                                  properties=properties,
                                  extra_properties=(form_factor, volume, mean_phase_shift, mch))
    return prop_dict
