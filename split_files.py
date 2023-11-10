import pandas as pd
import os
import random
import shutil

dataset = 'UDWA'

if dataset == 'CVRP':
    source_reference_folder = "./CVPR_subset"
    source_query_folder = "./CVPR_subset"
    destination_train_reference_folder = "./CVPR_subset/Train/reference_images/"
    destination_train_query_folder = "./CVPR_subset/Train/query_images/"
    destination_test_reference_folder = "./CVPR_subset/Test/reference_images/"
    destination_test_query_folder = "./CVPR_subset/Test/query_images/"

    split_files = ["train.csv", "test.csv"]
    for split_file in split_files:
        df = pd.read_csv("./CVPR_subset/splits/" + split_file, header=None)
        print(df.head())
        for index, row in df.iterrows():
            reference_image_path = os.path.join(source_reference_folder, row.iloc[0])
            query_image_path = os.path.join(source_query_folder, row.iloc[1])

            print(reference_image_path)
            if split_file == "train.csv":
                shutil.copy(reference_image_path, destination_train_reference_folder)
                shutil.copy(query_image_path, destination_train_query_folder)
            else:
                shutil.copy(reference_image_path, destination_test_reference_folder)
                shutil.copy(query_image_path, destination_test_query_folder)
elif dataset == 'UDWA':
    source_directory = './UDWA/total'

    train_reference_directory = './UDWA/Train/reference_images'
    train_query_directory = './UDWA/Train/query_images'
    test_reference_directory = './UDWA/Test/reference_images'
    test_query_directory = './UDWA/Test/query_images'

    train_ratio = 0.3
    test_ratio = 0.7

    png_files = [f for f in os.listdir(source_directory) if f.endswith('.png')]

    random.shuffle(png_files)

    total_files = len(png_files)
    num_train = int(total_files * train_ratio)
    num_test = total_files - num_train

    train_files = png_files[:num_train]
    test_files = png_files[num_train:]

    for file in train_files:
        file = file[:-4]

        source_ref_path = os.path.join(source_directory, file + '.png')
        destination_ref_path = os.path.join(train_reference_directory, file + '.png')

        source_query_path = os.path.join(source_directory, file + '.jpg')
        destination_query_path = os.path.join(train_query_directory, file + '.jpg')

        shutil.copy(source_ref_path, destination_ref_path)
        shutil.copy(source_query_path, destination_query_path)

    for file in test_files:
        file = file[:-4]

        source_ref_path = os.path.join(source_directory, file + '.png')
        destination_ref_path = os.path.join(test_reference_directory, file + '.png')

        source_query_path = os.path.join(source_directory, file + '.jpg')
        destination_query_path = os.path.join(test_query_directory, file + '.jpg')

        shutil.copy(source_ref_path, destination_ref_path)
        shutil.copy(source_query_path, destination_query_path)
