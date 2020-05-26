import numpy as np
import pandas as pd
import cv2
from typing import Dict, List
import json
import re
from pathlib import Path

def get_tile_select_params(param_path: str = 'optimal_tile_params.json'):
    with open(param_path, 'r') as fp:
        params = json.load(fp)
    return params

def get_config(config_path: str = 'config.json') -> Dict:
    """ Loads local path configurations

    Keyword Arguments:
        config_path {str} -- path to config json file (default: {'config.json'})

    Returns:
        Dict -- loaded configurations
    """
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    required_config_options = [
        'limbal_img_dir',
        'bcd_dir',
        'bcd_metadata'
    ]
    for opt in required_config_options:
        if opt not in config.keys():
            raise ValueError('Required config options missing: ' + opt)
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
    Assumes file structure from downloaded zip file intact.
    Requires 'bcd_dir' to be set in config.

    Arguments:
        img_class {str} -- Image class 'Control', 'Moderate', etc.
        patient_id {str} -- Patient id example: 'N1-OD-1'
        slice_id {int} -- integer slice number

    Raises:
        ValueError: Invalid image class
        ValueError: Invalid patient id
        ValueError: Invalid slice id

    Returns:
        cv2.image
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

    if int(slice_id) < 10:
        slice_id = '0' + str(slice_id)

    img_path = Path(class_dir, patient_id, '-' + str(slice_id) + '.jpg')
    if not img_path.is_file():
        raise ValueError('Image slice does not exist: ' + str(img_path))

    img = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    if img.shape != (384, 384):
        print('Warning, image not 384x384 pixels as standard')
    return img


def is_subject_format(subject):
    match = re.match(r'D\d+\sR\d+\s*', subject)
    if match is None:
        return False
    return True


def get_lsc_subjects():
    config = get_config()
    img_dir = Path(config['limbal_img_dir'])
    subjects = []
    for folder in img_dir.iterdir():
        if is_subject_format(folder.name):
            subjects.append(folder.name.split(' day')[0])
    return subjects


def get_lsc_image(
    subject: str,
    image_num: int = 1,
    day: int = 9,
    get_tile: bool = False,
):
    if not is_subject_format(subject):
        raise ValueError(subject + ' subject not in pattern D# R#')

    config = get_config()
    img_dir = Path(config['limbal_img_dir'])
    subject_dir = Path(
        img_dir,
        subject.strip() + ' day ' + str(day)
    )
    if not subject_dir.is_dir():
        raise ValueError(
            'Subject dir does not exist: '
            + str(subject_dir)
        )

    if get_tile:
        new_im_dir = Path(subject_dir, 'new')
        if new_im_dir.is_dir():
            img_path = Path(
                new_im_dir,
                'Image' + str(image_num) + '.tif'
            )
        else:
            img_path = Path(
                subject_dir,
                'Image' + str(image_num) + '_dots.tif'
            )
    else:
        img_path = Path(subject_dir, 'Image' + str(image_num) + '.jpg')
    if img_path.is_file():
        return cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    else:
        print('Warning image not found:', img_path)
        return None

def find_image_nums_per_subject(subject, day = 9):
    config = get_config()
    img_dir = Path(config['limbal_img_dir'])
    subject_dir = Path(
        img_dir,
        subject.strip() + ' day ' + str(day)
    )
    if not subject_dir.is_dir():
        raise ValueError('Subject dir does not exist: ' + str(subject_dir))

    image_nums = []
    for item in subject_dir.glob('Image*.jpg'):
        match = re.findall(r'\d+', item.name)
        image_nums.append(int(match[0]))
    return image_nums


def get_all_lsc_images(
    get_tile: bool = False,
):
    metadata = {
        'image_num': [],
        'subject': [],
    }
    images = []
    subjects = get_lsc_subjects()

    for subject in subjects:
        for img_num in find_image_nums_per_subject(subject):
            img = get_lsc_image(
                subject=subject,
                image_num=img_num,
                get_tile=get_tile
            )
            if img is not None:
                metadata['image_num'].append(img_num)
                metadata['subject'].append(subject)
                images.append(img)
    metadata = pd.DataFrame.from_dict(metadata)
    if metadata.shape[0] != len(images):
        raise ValueError('Some images not loaded, metadata generated does not match loaded images')
    return images, metadata
