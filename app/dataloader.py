import os
import json
import shutil
import zipfile

from PIL import Image
from tqdm import tqdm
from sentence_transformers import util

def get_data_zipweb(config: dict, zip_filename: str):
    print('Downloading images from web...')
    # creates data folder
    if not os.path.exists(config['data_folder']):
        os.makedirs(config['data_folder'], exist_ok=True)
    data_file_path = os.path.join(config['data_folder'], zip_filename)
    # images are contained in a zip file that can be downloaded from the web
    if not os.path.exists(data_file_path):
        util.http_get(config['data_url'], data_file_path)
    
    unzip_data_path = '.'.join(data_file_path.split('.')[:-1])
    if not os.path.exists(unzip_data_path):
        # unzip data
        os.makedirs(unzip_data_path)
        with zipfile.ZipFile(data_file_path, 'r') as zf:
            for member in tqdm(zf.infolist(), desc='Extracting'):
                zf.extract(member, unzip_data_path)
    print(f'Loaded {len(os.listdir(unzip_data_path))} items')
    
    return unzip_data_path


def mobile_desktop_split(config: dict, images_all_folder_path: str):
    print('Splitting images by device...')
    # start with empty folders
    for device in ['mobile', 'desktop']:
        if os.path.exists(config[device]['images_folder']['path']):
            shutil.rmtree(config[device]['images_folder']['path'])
        os.makedirs(config[device]['images_folder']['path'])
    # check aspect ratio of each image
    for file in tqdm(os.listdir(images_all_folder_path)):
        im = Image.open(os.path.join(images_all_folder_path, file))
        width, height = im.size
        target_device = 'desktop' if width >= 1.2*height else 'mobile'
        shutil.copy(
            os.path.join(images_all_folder_path, file),
            os.path.join(config[target_device]['images_folder']['path'], file)
        )
    print('Downloaded {} mobile images and {} desktop images'.format(len(os.listdir(config['mobile']['images_folder']['path'])), len(os.listdir(config['desktop']['images_folder']['path']))))
    return True



if __name__ == '__main__':
    # load config file
    PROJECT_CONFIG = 'config_unsplash.json'
    config = json.load(open(PROJECT_CONFIG))
    for device in ['mobile', 'desktop']:
        config[device]['images_folder']['path'] = os.path.join(config['data_folder'], config[device]['images_folder']['filename'])
    # download zip and extract all images
    images_all_folder_path = get_data_zipweb(config, 'unsplash-25k-photos.zip')
    # split by device
    mobile_desktop_split(config, images_all_folder_path)
