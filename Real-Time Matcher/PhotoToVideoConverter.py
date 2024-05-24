import cv2
import os

image_folder = ''
video_name = ''

images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

print(images[0])
frame = cv2.imread(images[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 16, (width, height))

for image in images:
    video.write(cv2.imread(image))
video.release()