import numpy as np
import pandas as pd
import json


def get_coordinates(file_path, dataset):
    if dataset == "CVPR":
        data = pd.read_csv(file_path, header=None)

        x1, y1 = np.array(data.iloc[:, 0].tolist()), np.array(data.iloc[:, 1].tolist())  # bottom left corner of bbox
        x2, y2 = np.array(data.iloc[:, 2].tolist()), np.array(data.iloc[:, 3].tolist())  # top right corner of bbox
        return x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2  # getting center of image

    if dataset == "UDWA":
        f = open(file_path)
        data = json.load(f)

        return data["longitude"], data["latitude"]

    if dataset == 'DenseUAV':
        with open(file_path, "r") as file:
            for line in file:
                # Split the line by spaces
                parts = line.split()
                # Extract the E and N coordinates
                return parts[1][1:], parts[2][1:]
