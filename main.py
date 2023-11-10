import os
import shutil
import random
from get_arial_image import get_image
from get_coordinates import get_coordinates

coordinates_path = 'C:\\Users\\lordres\\Downloads\\place'  # '.\\CVPR_subset\\annotations.csv'
reference_path = '.\\UDWA\\total'  # '.\\CVPR_subset\\reference_images'

dataset = 'UDWA'

if not os.path.exists(reference_path):
    os.mkdir(reference_path)

if dataset == 'CVPR':
    latitudes, longitudes = get_coordinates(coordinates_path, dataset)
    for i, (latitude, longitude) in enumerate(zip(latitudes, longitudes)):
        image = get_image(latitude, longitude, 19.1, i)
        if image:
            with open(f'{reference_path}\\satellite_image{i + 1}.png', 'wb') as f:
                f.write(image)
elif dataset == 'UDWA':
    img_idx = 0
    for i in range(21, 22):
        folder_path = coordinates_path + str(i)
        file_list = os.listdir(folder_path)

        json_files = [file for file in file_list if file.endswith('.json')]
        jpg_files = [file for file in file_list if file.endswith('.jpg')]

        for json_file in json_files:
            img_idx += 1
            file_path = os.path.join(folder_path, json_file)
            longitude, latitude = map(float, get_coordinates(file_path, dataset))
            image = get_image(str(latitude + random.randint(-10, 10) / 30000),
                              str(longitude + random.randint(-10, 10) / 30000), 19, i)

            if image:
                with open(f'{reference_path}\\{json_file[:-5]}.png', 'wb') as f:
                    f.write(image)

        for jpg_file in jpg_files:
            source_file_path = os.path.join(folder_path, jpg_file)
            destination_file_path = os.path.join(reference_path, jpg_file)

            shutil.move(source_file_path, destination_file_path)
