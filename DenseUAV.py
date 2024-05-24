import os
import random
from enum import Enum
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2 as cv
import numpy as np
import albumentations as A

import torch
class Mode(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


GEOMETRIC_DOUBLE = 'GEOMETRIC_DOUBLE'
GEOMETRIC_SINGLE = 'GEOMETRIC_SINGLE'
FINE_SINGLE = 'FINE_SINGLE'
COLOR_DOUBLE = 'COLOR_DOUBLE'
COLOR_SINGLE = 'COLOR_SINGLE'
RANDOM_CROP_SINGLE = 'RANDOM_CROP_SINGLE'
RANDOM_CROP_DOUBLE = 'RANDOM_CROP_DOUBLE'


def make_train_aug(size=(512, 512)):
    h, w = size
    geometric_aug = [
        A.Flip(p=0.75),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.75),
        A.ShiftScaleRotate(scale_limit=(-0.5, 0.5), shift_limit=0, rotate_limit=45, p=0.9),
        A.Perspective(p=0.25),
        A.PadIfNeeded(min_height=h, min_width=w, always_apply=True, border_mode=0),
    ]

    geometric_double = A.Compose(geometric_aug, additional_targets={'positive': 'image'})

    color_aug = [
        A.Sharpen(alpha=(0.05, 0.1), lightness=(0.1, 0.5), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
        A.RGBShift(),
        A.CLAHE(p=0.2),
        A.RandomGamma(p=1),
        A.HueSaturationValue(p=1),
        A.ChannelShuffle(p=0.2),

        A.OneOf([
            A.GaussNoise(p=1),
            A.Emboss(p=1),
            A.Sharpen(p=1),
            A.ImageCompression(p=1),
        ], p=0.75),

        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.GaussianBlur(blur_limit=3, p=1),
            A.MedianBlur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.75),
    ]

    color_double = A.Compose(color_aug, additional_targets={'negative': 'image'})

    return {
        GEOMETRIC_DOUBLE: geometric_double,
        COLOR_DOUBLE: color_double,
    }

class DenseUAVDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.is_debug = False

        if self.mode == Mode.VAL:
            self.root += '/test/'
            self.query_path = 'query_drone/'
            self.ref_path = 'gallery_satellite/'
        else:
            self.root += '/train/'
            self.query_path = 'drone/'
            self.ref_path = 'satellite/'

        sorted(os.listdir(self.root + self.query_path))
        self.query_folders = sorted(os.listdir(self.root + self.query_path))
        self.ref_folders = sorted(os.listdir(self.root + self.ref_path))

        self.query_img_path = []
        self.ref_img_path = []

        if self.mode == Mode.TRAIN:

            for folder_name in self.query_folders:
                folder_path = os.path.join(self.root + self.query_path, folder_name)
                self.query_img_path.extend([
                    os.path.join(folder_path, f"H100.JPG"),
                    # os.path.join(folder_path, f"H90.JPG"),
                    # os.path.join(folder_path, f"H80.JPG")
                ])

            for folder_name in self.ref_folders:
                folder_path = os.path.join(self.root + self.ref_path, folder_name)
                self.ref_img_path.extend([
                    os.path.join(folder_path, f"H100.tif") #if random.randint(0, 1) == 0 else os.path.join(
                        #folder_path, f"H100_old.tif"),
                    # os.path.join(folder_path, f"H90_old.tif") if random.randint(0, 1) == 0 else os.path.join(
                    #     folder_path, f"H90_old.tif"),
                    # os.path.join(folder_path, f"H80.tif") if random.randint(0, 1) == 0 else os.path.join(folder_path,
                    #                                                                                      f"H80_old.tif")
                ])

        else:

            self.ref_folders = sorted(os.listdir(self.root + self.ref_path))[int(self.query_folders[0]):]

            for folder_name in self.query_folders:
                folder_path = os.path.join(self.root + self.query_path, folder_name)
                self.query_img_path.append(os.path.join(folder_path, f"H100.JPG"))

            for folder_name in self.ref_folders:
                folder_path = os.path.join(self.root + self.ref_path, folder_name)
                self.ref_img_path.append(os.path.join(folder_path, f"H100.tif"))

    def get_image(self, path):
        image = Image.open(path)

        return image

    def generate_ref_image(self, index):
        positive_folder = self.ref_img_path[index]
        rand_idx = random.randint(0, len(self.ref_img_path) - 1)  # 0, 1, 2 - 3, 4 ... +  3, 4, 5 - 0, 1, 2, 6, 7 , 8
        valid_range_start = (index // 3) * 3
        valid_range_end = valid_range_start + 2
        while rand_idx > valid_range_start and rand_idx < valid_range_end:  # 3826 1289
            rand_idx = random.randint(0, len(self.ref_img_path) - 1)
        negative_folder = self.ref_img_path[rand_idx]
        return positive_folder, negative_folder, torch.as_tensor(rand_idx // 3, dtype=torch.int)

    def apply_color_transfer(self, image_target, image_source):
        mean_target, mean_source = np.mean(image_target, axis=(0, 1)), np.mean(image_source, axis=(0, 1))
        std_target, std_source = np.std(image_target, axis=(0, 1)), np.std(image_source, axis=(0, 1))

        colored_image = (image_source - mean_source) * (std_target / std_source) + mean_target
        colored_image = np.clip(colored_image, 0, 255).astype(np.uint8)
        return colored_image

    def custom_transform(self, image, is_ref=False):

        image = transforms.ToTensor()(image)
        image = transforms.Resize((224, 224))(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        return image

    def __getitem__(self, index):
        anchor_label, positive_label = torch.as_tensor(index // 3, dtype=torch.int), torch.as_tensor(index // 3,
                                                                                                     dtype=torch.int)
        anchor = self.get_image(self.query_img_path[index])
        positive_path, negative_path, negative_label = self.generate_ref_image(index)

        positive = self.get_image(positive_path)
        negative = self.get_image(negative_path)

        if self.mode == Mode.TRAIN:
            sample = make_train_aug()[GEOMETRIC_DOUBLE](image=cv.resize(np.array(anchor), (512, 512)),
                                                        positive=np.array(positive))
            anchor, positive = sample['image'], sample['positive']

            sample = make_train_aug()[COLOR_DOUBLE](image=np.array(positive), negative=np.array(negative))
            positive, negative = sample['image'], sample['negative']

        if not self.is_debug:
            anchor = self.custom_transform(anchor)
            positive = self.custom_transform(positive, is_ref=True)
            negative = self.custom_transform(negative, is_ref=True)

        return anchor, positive, negative, anchor_label, positive_label, negative_label

    def __len__(self):
        return len(self.query_img_path)