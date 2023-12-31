import cv2 as cv
import numpy as np
import torch
from torch import from_numpy
from torchvision import transforms
from skimage.exposure import match_histograms

class Resize:
    def __init__(self, output_size: (int, int)):
        self.output_size = output_size

    def __call__(self, sample):
        query_image, reference_image = sample['query'], sample['reference']

        resized_query_image = cv.resize(query_image, self.output_size)
        resized_reference_image = cv.resize(reference_image, self.output_size)

        return {'query': resized_query_image, 'reference': resized_reference_image}


class Normalize:
    def __init__(self, interval: (int, int)):
        assert interval[0] < interval[1]
        self.interval = interval

    def __call__(self, sample):
        query_image, reference_image = sample['query'], sample['reference']

        normed_query_image = cv.normalize(query_image, None, alpha=self.interval[0],
                                          beta=self.interval[1], norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        normed_reference_image = cv.normalize(reference_image, None, alpha=self.interval[0],
                                              beta=self.interval[1], norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        return {'query': normed_query_image, 'reference': normed_reference_image}


class ToTensor(object):
    def __call__(self, sample):
        query_image, reference_image = sample['query'], sample['reference']

        query_image = query_image.transpose((2, 0, 1))
        reference_image = reference_image.transpose((2, 0, 1))

        return {'query': from_numpy(query_image), 'reference': from_numpy(reference_image)}

class MeanStdRecorder():
    def __init__(self, images):
        rgb_values = np.concatenate(images, axis=0) / 255
        rgb_values = np.reshape(rgb_values, (-1, 3))

        self.means = np.mean(rgb_values, axis=0)
        self.stds = np.std(rgb_values, axis=0)
        self.len = images.shape[0]

    def update(self, images):
        rgb_values = np.concatenate(images, axis=0) / 255
        rgb_values = np.reshape(rgb_values, (-1, 3))

        new_means = np.mean(rgb_values, axis=0)
        new_stds = np.std(rgb_values, axis=0)

        n = images.shape[0]
        m = self.len

        tmp = self.means

        self.means = m / (m + n) * tmp + n / (m + n) * new_means
        self.stds = m / (m + n) * self.stds**2 + n / (m + n) * new_stds**2 + m * n / (m + n)**2 * (tmp - new_means)**2
        self.stds = np.sqrt(self.stds)

        self.len += n


class CustomTransform:
    def __init__(self, q_mean, q_std, ref_mean, ref_std):
        self.query_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=q_mean, std=q_std),
            transforms.ToTensor()
        ])

        self.reference_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=ref_mean, std=ref_std),
            transforms.ToTensor(),
        ])

    def __call__(self, sample):
        query_image = sample['query']
        reference_image = sample['reference']

        query_image = self.query_transform(query_image)
        reference_image = self.reference_transform(reference_image)

        return {'query': query_image, 'reference': reference_image}