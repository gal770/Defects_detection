import numpy as np
from PIL import Image
import os
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

""" This is a script for creating patches of clean and defective images,
 and dividing them into train, val and test datasets."""



def SaveToDestination(des_path, img_sub, img, blobs_img, bounds, col, counter):
    """ Function which checks weather sample is clean or defective, and divides into train, val and test datasets."""
    if col <= 14:
        part = 'train'
    elif col <= 17:
        part = 'val'
    elif col <= 20:
        part = 'test'
    blobs_arr = np.array(blobs_img.crop(bounds))
    if np.max(blobs_arr) == 0:
        (img.crop(bounds)).save(os.path.join(os.getcwd(), des_path, '%s/clean/%s%d.tiff' % (part, img_sub, counter)))
        (blobs_img.crop(bounds)).save(
            os.path.join(os.getcwd(), des_path, '%s/clean/%s%d_blobs.tiff' % (part, img_sub, counter)))
    else:
        (img.crop(bounds)).save(os.path.join(os.getcwd(), des_path, '%s/defect/%s%d.tiff' % (part, img_sub, counter)))
        (blobs_img.crop(bounds)).save(
        os.path.join(os.getcwd(), des_path, '%s/defect/%s%d_blobs.tiff' % (part, img_sub, counter)))



def AnalyseDatafromPic(root_path, crop_size, col, line, des_path):
    for column in range(0, col + 1):
        for lin in range(0, line + 1):
            counter = 0
            print('line: %d, column: %d' % (lin, column))
            images_sub = f'{column:03d}' + '_' f'{lin:03d}' + '_'
            zone_img = Image.open(os.path.join(os.getcwd(), root_path, "%sZone.tif" % images_sub))
            zone_img = zone_img.convert('RGB')
            img = Image.open(os.path.join(os.getcwd(), root_path, "%sInsp.tif" % images_sub))
            blobs_img = Image.open(os.path.join(os.getcwd(), root_path, "%sGSI12_Zone_0_FinalBlobs.tif" % images_sub))
            for i in range(0, 1792, crop_size):
                for j in range(0, 1792, crop_size):
                    bounds = (j, i, j + crop_size, i + crop_size)
                    crop_zone = zone_img.crop(bounds)
                    crop_zone = np.sum(np.array(crop_zone), axis=2)
                    #Checking weather crop is relevant
                    if np.min(np.array(crop_zone)) > 0:
                        SaveToDestination(des_path, images_sub, img, blobs_img, bounds, column, counter)
                    counter += 1


def balance_train_val_datasets(des_path):
    """ Function that check which label (clean/defect) contains more training data,
    and balances between both datasets."""

    train_clean_images = os.listdir(os.path.join(des_path, 'train/clean'))
    train_defect_images = os.listdir(os.path.join(des_path, 'train/defect'))
    val_clean_images = os.listdir(os.path.join(des_path, 'val/clean'))
    val_defect_images = os.listdir(os.path.join(des_path, 'val/defect'))
    clean_train_len = len(train_clean_images)
    defect_train_len = len(train_defect_images)
    if clean_train_len < defect_train_len:
        train_len = clean_train_len
    else:
        train_len = defect_train_len

    clean_val_len = len(val_clean_images)
    defect_val_len = len(val_defect_images)
    if clean_val_len < defect_val_len:
        val_len = clean_train_len
    else:
        val_len = defect_train_len

    random.shuffle(train_clean_images)
    random.shuffle(train_defect_images)
    counter = 0
    for image in train_clean_images:
        if counter >= train_len:
            os.remove(os.path.join(des_path, 'train/clean', image))
        counter += 1
    counter = 0
    for image in train_defect_images:
        if counter >= train_len:
            os.remove(os.path.join(des_path, 'train/defect', image))
        counter += 1
    random.shuffle(val_clean_images)
    random.shuffle(val_defect_images)
    counter = 0
    for image in val_clean_images:
        if counter >= val_len:
            os.remove(os.path.join(des_path, 'val/clean', image))
        counter += 1
    counter = 0
    for image in val_defect_images:
        if counter >= val_len:
            os.remove(os.path.join(des_path, 'val/defect', image))
        counter += 1


if __name__ == '__main__':
    crop_size = 128
    AnalyseDatafromPic("AD_CTW_X5", crop_size, 20, 14,
                       "AD_CTW_%d" % crop_size)

    # Ensuring training and val datasets are balanced
    balance_train_val_datasets("AD_CTW_%d" % crop_size)

