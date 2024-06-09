import os
import random
from shutil import copy2

from PIL import Image
import numpy as np
from patchify import patchify
from tqdm import tqdm

from enums.label import Label
from utils.logvisor import logger


def check_is_dir(path: str) -> bool:
    if not os.path.isdir(path):
        raise ValueError(f"Provided path: {path} is not a directory")
    logger.info(f"The direction : {path} exist.")
    return True


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        logger.info(f"Created directory : {path}")
    else:
        logger.info(f"Directory -> {path} already exist")


def find_files(directory, extension):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


def crop_to_fit(img, patch_size):
    new_width = (img.width // patch_size) * patch_size
    new_height = (img.height // patch_size) * patch_size
    return img.crop((0, 0, new_width, new_height))


def patch_and_save(file_paths, save_path, patch_size):
    for file_path in file_paths:
        img = Image.open(file_path)
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # Tek kanallı (grayscale) görüntüler için
            patch_shape = (patch_size, patch_size)
        else:  # Çok kanallı (örneğin RGB) görüntüler için
            patch_shape = (patch_size, patch_size, img_array.shape[2])

        img = crop_to_fit(img, patch_size)
        original_size = img.size
        patches = patchify(img_array, patch_shape, step=patch_size)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        logger.info(f'Processing file: {file_path} | Original size: {original_size}')

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = Image.fromarray(patches[i, j, 0])
                patch_filename = f'{base_filename}_patch_{i}_{j}.png'
                patch.save(os.path.join(save_path, patch_filename))
                logger.info(f'Saved patch: {patch_filename} | Size: {patch.size} | Location: {save_path}')


def mask_3d_to_2d_labeled(masks_folder, save_dir):
    mask_files = find_files(masks_folder, '.png')
    create_directory(save_dir)

    for path in tqdm(mask_files):
        img = Image.open(path)
        img_array = np.array(img)

        label_mask = np.zeros(img_array.shape, dtype=np.uint8)

        label_mask[np.all(img_array == Label.BUILDING.hex_to_rgb(), axis=-1)] = 0  # Building
        label_mask[np.all(img_array == Label.LAND.hex_to_rgb(), axis=-1)] = 1  # Land
        label_mask[np.all(img_array == Label.ROAD.hex_to_rgb(), axis=-1)] = 2  # Road
        label_mask[np.all(img_array == Label.UNLABELED.hex_to_rgb(), axis=-1)] = 3  # Vegetation
        label_mask[np.all(img_array == Label.WATER.hex_to_rgb(), axis=-1)] = 4  # Water
        label_mask[np.all(img_array == Label.UNLABELED.hex_to_rgb(), axis=-1)] = 5  # Unlabeled

        save_path = os.path.join(save_dir, os.path.basename(path.split('.')[2]))
        np.save(save_path, label_mask[:, :, 0])


def process_datasets(images_folder, masks_folder, save_path_images, save_path_masks, patch_size=256):
    image_files = find_files(images_folder, '.jpg')
    mask_files = find_files(masks_folder, '.png')
    create_directory(save_path_masks)
    create_directory(save_path_images)
    patch_and_save(image_files, save_path_images, patch_size)
    patch_and_save(mask_files, save_path_masks, patch_size)


def select_random_files(images_dir, mask_dir, data_dir_images, data_dir_masks, num_val_test=50):
    all_images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    all_masks = [f for f in os.listdir(mask_dir) if f.endswith('.npy')]

    random.shuffle(all_images)

    val_images = all_images[:num_val_test]
    test_images = all_images[num_val_test:num_val_test * 2]
    train_images = all_images[num_val_test * 2:]

    for image in train_images:
        mask = image.replace('.png', '.npy')
        copy2(os.path.join(images_dir, image), os.path.join(data_dir_images, "train", image))
        copy2(os.path.join(mask_dir, mask), os.path.join(data_dir_masks, "train", mask))

    for image in val_images:
        mask = image.replace('.png', '.npy')
        copy2(os.path.join(images_dir, image), os.path.join(data_dir_images, "val", image))
        copy2(os.path.join(mask_dir, mask), os.path.join(data_dir_masks, "val", mask))

    for image in test_images:
        mask = image.replace('.png', '.npy')
        copy2(os.path.join(images_dir, image), os.path.join(data_dir_images, "test", image))
        copy2(os.path.join(mask_dir, mask), os.path.join(data_dir_masks, "test", mask))


image_directory = '../data/patched_images'
mask_directory = '../data/masks_2d'
data_dir_image = "../data/dataset/images"
data_dir_mask = "../data/dataset/masks"

select_random_files(image_directory, mask_directory, data_dir_image, data_dir_mask)

# mask_3d_to_2d_labeled("../data/patched_masks", "../data/masks_2d")

"""process_datasets('../Semantic Segmentation Dataset',
                 '../Semantic Segmentation Dataset',
                 'data/patched_images', 'data/patched_masks')"""
