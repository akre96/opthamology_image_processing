import numpy as np
import pandas as pd
import cv2
from typing import Dict, List
import json
from pathlib import Path

def get_config(config_path: str = 'config.json') -> Dict:
    """ Loads local path configurations

    Keyword Arguments:
        config_path {str} -- path to config json file (default: {'config.json'})

    Returns:
        Dict -- loaded configurations
    """
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config


def load_bcd_metadata(sheet_name: str = 'basal cell density'):
    """ Loads and cleans results from cell counting file

    Keyword Arguments:
        sheet_name {str} -- which sheet in the file to load
            (default: {'basal cell density'})

    Returns:
        pd.DataFrame -- cleaned dataframe of data
    """
    md_path = get_config()['bcd_metadata']
    metadata = pd.read_excel(
        md_path,
        sheet_name=sheet_name,
        header=1,
    )
    metadata.columns = [x.strip().lower() for x in metadata.columns]
    metadata['study_number_id_eye'] = metadata['study_number_id_eye']\
        .str.replace(' ', '')
    metadata['clinical'] = metadata['clinical']\
        .str.replace(' ', '')
    return metadata

def convert_id_to_folder_label(
    study_num_eye: str,
    cc_scan: str
) -> List:
    """ convert from pid-eye + scan-slice format to pid-eye-scan + slice 

    Arguments:
        study_num_eye {str} -- example: N1-OD-1
        cc_scan {str} -- example: 1-28

    Returns:
        List -- [pid-eye-scan, slice]
    """
    cc, img_slice = cc_scan.split('-')
    p_id = '-'.join([study_num_eye, cc])
    p_id = p_id.replace('-OD-', '-od-').replace('-OS-', '-os-')
    return p_id, img_slice



def get_bcd_image(
    img_class: str = None,
    patient_id: str = None,
    slice_id: int = 30,
) -> np.ndarray:
    """ read an image from BCD images

    Arguments:
        img_class {str} -- Image class 'Control', 'Moderate', etc.
        patient_id {str} -- Patient id example: 'N1-OD-1'
        slice_id {int} -- integer slice number

    Raises:
        ValueError: Invalid image class
        ValueError: Invalid patient id
        ValueError: Invalid slice id

    Returns:
        image
    """
    config = get_config()
    img_dir = Path(config['bcd_dir'])
    img_classes = ['Severe', 'Moderate', 'Mild', 'Control']

    if img_class not in img_classes:
        raise ValueError('img_class must be in ' + ', '.join(img_classes))
    class_dir = Path(img_dir, img_class)

    patient_ids = [
        x.name for x in class_dir.iterdir()
        if (x.is_dir() and (x.name != 'Anonymized'))
    ]

    if patient_id not in patient_ids:

        raise ValueError('Patient ID must be in: ' + ', '.join(patient_ids))

    img_path = Path(class_dir, patient_id, '-' + str(slice_id) + '.jpg')
    if not img_path.is_file():
        raise ValueError('Image slice does not exist: ' + str(img_path))

    return cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)

            
            

