
from io import BytesIO
from PIL import Image
from pyproj import Proj
import pickle
import numpy as np
import time

from get_arial_image import get_image


def convert_easting_northing_to_lat_long(easting, northing, zone_number):
    utm_projection = Proj(proj='utm', zone=zone_number, ellps='WGS84')

    longitude, latitude = utm_projection(easting, northing, inverse=True)

    return latitude, longitude


def create_satellite_map(latitude, longitude, offsetX, offsetY, zoom, scale, patch_shape, tile_shape, map_shape, crop_margin=0.2):
    def concatenate_images(num_rows, num_cols):
        num_images = len(images)
        metadata = []

        if num_rows * num_cols != num_images:
            raise ValueError("Number of images must match the grid dimensions.")

        image_width, image_height = images[0].size

        grid_width = image_width * num_cols
        grid_height = image_height * num_rows
        grid_image = Image.new('RGB', (grid_width, grid_height))

        for i in range(num_images):
            row_index = i // num_cols
            col_index = i % num_cols
            x_offset = col_index * image_width
            y_offset = row_index * image_height
            grid_image.paste(images[i], (x_offset, y_offset))
            metadata.append({
                'index': row_index * i + col_index,
                'row_index': row_index,
                'col_index': col_index,
                'longitude': longitude + row_index * long_step,
                'latitdue': latitude + col_index * lat_step
            })
        return metadata, grid_image

    w, h = patch_shape
    w_tile, h_tile = tile_shape
    map_rows, map_cols = map_shape
    w_cropped, h_cropped = w * (1 - crop_margin), h * (1 - crop_margin)

    long_step = (360 / 2 ** zoom) * ((w_cropped / 2) / w_tile)

    R = 6378137
    dn = 640 * 23.609
    de = 640 * 23.64
    lat_step = dn / R

    print(lat_step)
    print(long_step)

    latitude += offsetY * lat_step
    longitude += offsetX * long_step

    left = int((w - w_cropped) / 2)
    top = int((h - h_cropped) / 2)
    right = int((w + w_cropped) / 2)
    bottom = int((h + h_cropped) / 2)

    images = []
    start = time.time()
    for i in range(map_rows):
        for j in range(map_cols):
            image = BytesIO(get_image(latitude - i * lat_step, longitude + j * long_step, 18, 0))
            image = Image.open(image).convert("RGB").crop((left, top, right, bottom))
            images.append(image)
            print(f'#({i + 1}, {j + 1}) Image processed')

    end1 = time.time()
    print(f'All images processed. Estimated time: {end1 - start}')

    metadata, concatenated_image = concatenate_images(map_rows, map_cols)
    Image._show(concatenated_image)

    end2 = time.time()
    print(f'All images concatenated. Estimated time: {end2 - end1}\nOverall time: {end2 - start}; '
          f'overall shape: {concatenated_image.size}')
    save_data(metadata, 'metadata')
    save_images(images, 'images')
    return metadata, images, concatenated_image

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def save_images(images, filename):
    image_arrays = [np.array(image) for image in images]
    concatenated_array = np.stack(image_arrays)
    np.save(filename, concatenated_array)

def load_images(filename):
    concatenated_array = np.load(filename)
    images = [Image.fromarray(image_array) for image_array in concatenated_array]
    return images



def get_satellite_map():
    lat, long = 30.309400805555555, 120.34963647222222
    return create_satellite_map(lat, long, 0, 7, 18, 2, (1280, 1280), (256, 256), (7, 16))
