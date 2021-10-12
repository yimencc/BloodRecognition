import os
import sys
import time
import logging.config
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from cccode                     import image
# from Deeplearning.evaluate      import PredHandle
# from Deeplearning.util.models   import YoloV5Model
from Deeplearning.util.dataset  import (dataset_xml_from_annotations, StandardXMLContainer, DATA_ROOT)

timeStamp   =   time.strftime("%Y%m%d-%H%M%S")
nx          =   np.newaxis
ck          =   image.Check(False, False, False)
MODEL_PATH  =   "..\\data\\models"

# Logging Config ---------------------------------------------------------------------------------------
logging.config.fileConfig(".\\log\\config\\util.conf")
logger      =   logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Logging Config Ended ---------------------------------------------------------------------------------


def pred_show(inputs: list[list], fig_path: str, time_stamp: str, idx: int, display: bool, save_fig=False):
    """
    Evaluate the model accuracy, this func will plot several figures,
    each figure contain eight rbc recognition results, both for
    artifact annotation and deep learning annotation.
    """
    scale = 320/40
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    fig, axs = plt.subplots(2, 4, constrained_layout=True)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    total_n_label, total_n_preds = 0, 0
    for i, (input_img, bboxes, coord_2d) in enumerate(inputs):
        row, col = i // 4, i % 4
        ax = axs[row, col]
        ax.imshow(input_img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        coord_list = coord_2d[np.where(coord_2d.sum(-1) > 0)]
        for bbox in bboxes:
            x, y, w, h = [elm*320/40 for elm in bbox[:4]]
            rect = plt.Rectangle((x-w/2, y-h/2), width=w, height=h, lw=1.1,
                                 fill=False, color="blue")
            ax.add_patch(rect)
        for coord in coord_list:
            x1, y1, x2, y2 = [elm*scale for elm in coord]
            rect = plt.Rectangle((x1, y1), width=(x2-x1), height=(y2-y1), lw=1.1, fill=False, color="red", alpha=.8)
            ax.add_patch(rect)
        n_label = len(bboxes)
        n_pred = len(coord_list)
        total_n_label += n_label
        total_n_preds += n_pred
        ax.set_xlabel(f"Label: {n_label} Preds: {n_pred}, Error: "
                      f"{abs(n_pred-n_label)/n_label*100:.2f}%", fontsize=8)
    fig.suptitle(f"Total Error: {abs(total_n_preds-total_n_label)/total_n_label*100:.2f}%", fontsize=12)
    if save_fig:
        plt.savefig(os.path.join(fig_path, "%s_%d.png" % (time_stamp, idx)))
    if display:
        plt.show()


def dataset_construct():
    """ Date: 2021-09-09
    Constructing dataset from generated modalities and automatic labeled annotations,
    for subsequent manual data annotating and network training. """
    sources_root    =   "D:\\Database\\prism_dual-tie_dataset\\20210902 RBC Detection"
    targets_root    =   "D:\\Workspace\\RBC Recognition\\datasets\\20210902 BloodSmear01"

    minus_path      =   join(sources_root, "blood smear 01\\minus")
    plus_path       =   join(sources_root, "blood smear 01\\plus")
    focus_path      =   join(sources_root, "blood smear 01\\focus")
    phase_path      =   join(sources_root, "blood smear 01\\phase")
    ann_path        =   join(sources_root, "blood smear 01\\annotation")

    dstXML_filename =   join(targets_root, "BloodSmear20210902_01.xml")
    container       =   dataset_xml_from_annotations(minus_path, plus_path, focus_path,
                                                     phase_path, ann_path, dstXML_filename)

    yolo_path       =   join(targets_root, "yolo")
    container.toyolo(yolo_path)


def image_splitting_test():
    """ Date: 2021-09-11 """
    from Deeplearning.util.dataset import StandardXMLContainer
    source_root     =   "D:\\Workspace\\RBC Recognition\\datasets"
    ffov_xml        =   join(source_root, "20210902 BloodSmear01", "BloodSmear20210902_01.xml")
    split_root      =   join(source_root, "20210902 BloodSmear01", "CVAT SourceData-SplitSamples")

    xml_docs        =   StandardXMLContainer.fromXML(ffov_xml)
    xml_docs.sample_slicing(split_root=split_root, n_batch=8)


def issue_20211002_01():
    """
    Because of the filenames inside the annotation XML files don't matched the accurate filepath
    of exact file after the original files is moved, the file reader couldn't found the files according
    to current given path. The file path recording should be corrected.
    """
    # Read the XML file
    from xml.etree import ElementTree as ET
    dstXmlFilename  =   join(DATA_ROOT, "20210105 BloodSmear\\fov_annotations.xml")

    etree = ET.parse(dstXmlFilename)
    stdContainer = StandardXMLContainer()
    stdContainer.root = etree.getroot()

    err_strings = "data\\2021-01-05"
    acc_folder  = "datasets\\20210105 BloodSmear"
    for sample in stdContainer.root:
        logger.debug(f"This item in stdContainer is a {type(sample)}")
        for tag in ["amp_fullname", "pha_fullname", "over_fullname", "under_fullname"]:
            element = sample.find(tag)
            try:
                assert err_strings in element.text
                element.text = element.text.replace(err_strings, acc_folder)
                assert os.path.isfile(element.text)
            except Exception as e:
                logger.exception(e)

    xmlStrings = ET.tostring(stdContainer.root, encoding="utf-8")
    with open(dstXmlFilename, "wb") as f:
        f.write(xmlStrings)


if __name__ == "__main__":
    """Run Entrance"""
    print(sys.version_info, "\n", sys.version)
    issue_20211002_01()
