import cv2 as cv
import numpy as np
import torch
import os
from preprocessing import Normalize
path_q = 'C:\\Users\\lordres\\Desktop\\Dyplom\\ALTO\\Train\\query_images\\'
path_ref = 'C:\\Users\\lordres\\Desktop\\Dyplom\\ALTO\\Train\\reference_images\\'
all_q_images = os.listdir(path_q)
# query_images = np.array(list(map(cv.imread, [path_q + q for q in os.listdir(path_q)[:16]])))
# ref_images = np.array(list(map(cv.imread, [path_ref + ref for ref in os.listdir(path_q)[:16]])))
#


# for i in range(3):
#     mean[i] += query_images[:, :, :, i].mean()
#     std[i] += query_images[:, :, :, i].std()
#
# mean.div_(len(query_images))
# std.div_(len(query_images))
#
# print(mean, std)
batch = 10
q_imgs = [path_q + q for q in all_q_images]

rgb_values = np.concatenate([cv.imread(img) for img in q_imgs[:batch]], axis=0)
rgb_values = np.reshape(rgb_values, (-1, 3))
mean = np.mean(rgb_values, axis=0)
std = np.std(rgb_values, axis=0)

print(mean.shape, std.shape)

t = Normalize(np.array([cv.imread(img) for img in q_imgs[:batch]]))
print(t.stds, t.means)

for i in range(batch, len(q_imgs), batch):
    t.update(np.array([cv.imread(img) for img in q_imgs[i:i + batch]]))
    if i % 10 == 0:
        print(t.stds, t.means)
