"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
import json

def RGBtoOneHot(rgb, colorList):
  arr = np.zeros(rgb.shape[:2], dtype=np.uint8)
  for i in range(len(colorList)):
    color = np.array(colorList[i]["color"], dtype=np.uint8)
    classid = colorList[i]["classid"]
    color = color[::-1]
    mask = np.all(rgb==color, axis=-1)
    arr[mask] = classid
  return arr

# data_path = Path('data')
data_path = Path('/content/drive/MyDrive/EndoVis2018')

train_path = data_path / 'train'

cropped_train_path = data_path / 'processed_train'

f = open('/content/drive/MyDrive/EndoVis2018/train/miccai_challenge_release_2/labels.json').read()
labels = json.loads(f)

# original_height, original_width = 1080, 1920
height, width = 1024, 1280
# h_start, w_start = 28, 320

binary_factor = 255
parts_factor = 85
instrument_factor = 32


if __name__ == '__main__':
    for instrument_index in range(1, 16):
        instrument_folder = 'seq_' + str(instrument_index)

        (cropped_train_path / instrument_folder / 'images').mkdir(exist_ok=True, parents=True)

        binary_mask_folder = (cropped_train_path / instrument_folder / 'binary_masks')
        binary_mask_folder.mkdir(exist_ok=True, parents=True)

        parts_mask_folder = (cropped_train_path / instrument_folder / 'parts_masks')
        parts_mask_folder.mkdir(exist_ok=True, parents=True)

        instrument_mask_folder = (cropped_train_path / instrument_folder / 'instruments_masks')
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)

        # mask_folders = list((train_path / instrument_folder / 'labels').glob('*.'))
        mask_folder=(train_path / instrument_folder / 'labels')
        # mask_folders = [x for x in mask_folders if 'Other' not in str(mask_folders)]

        for file_name in tqdm(list((train_path / instrument_folder / 'left_frames').glob('*.png'))):
            img = cv2.imread(str(file_name))
            old_h, old_w, _ = img.shape

            # img = img[h_start: h_start + height, w_start: w_start + width]
            cv2.imwrite(str(cropped_train_path / instrument_folder / 'images' / (file_name.stem + '.jpg')), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

            mask_binary = np.zeros((old_h, old_w), dtype=np.uint8)
            mask_parts = np.zeros((old_h, old_w), dtype=np.uint8)
            mask_instruments = np.zeros((old_h, old_w), dtype=np.uint8)

            
            labl = cv2.imread(str(mask_folder / file_name.name))
            mask = RGBtoOneHot(labl, labels)


            #binary
            mask_binary[mask > 0] = 1
            mask_binary[mask == 4] = 0
            mask_binary[mask == 5] = 0
            mask_binary[mask == 10] = 0

            #parts
            mask_parts[mask == 1] = 1  # Shaft
            mask_parts[mask == 2] = 2  # Wrist
            mask_parts[mask == 3] = 3  # Claspers

            #instruments
            mask_instruments[mask == 6] = 1
            mask_instruments[mask == 7] = 2
            mask_instruments[mask == 8] = 3
            mask_instruments[mask == 9] = 4
            mask_instruments[mask == 11] = 5
            mask_instruments[mask == 1] = 6  # Shaft
            mask_instruments[mask == 2] = 6  # Wrist
            mask_instruments[mask == 3] = 6  # Claspers
            
            mask_binary=mask_binary* binary_factor
            mask_parts=mask_parts* parts_factor
            mask_instruments=mask_instruments* instrument_factor
            cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
            cv2.imwrite(str(parts_mask_folder / file_name.name), mask_parts)
            cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)
