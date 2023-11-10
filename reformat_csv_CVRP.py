import csv


def replace_zeros(filename):
    if filename.startswith("bingmap"):
        filename_parts = filename.split("/")
        filename_parts[-1] = "satellite_image" + filename_parts[-1].lstrip('0')
        modified_name = "/".join(filename_parts)
        return replace_extension(modified_name)
    return filename


def replace_panos(filename):
    return filename.replace("streetview/panos", "query_images")


def replace_bingmap(filename):
    return filename.replace("bingmap/19", "reference_images")


def replace_extension(filename):
    return filename.replace(".jpg", ".png")


if __name__ == '__main__':
    input_csv_files = ["./CVPR_subset/splits/val-19zl.csv", "./CVPR_subset/splits/train-19zl.csv"]
    output_csv_files = ["./CVPR_subset/splits/test.csv", "./CVPR_subset/splits/train.csv"]

    for input_csv_file, output_csv_file in zip(input_csv_files, output_csv_files):
        with open(input_csv_file, mode="r") as input_file, open(output_csv_file, mode="w", newline="") as output_file:
            csv_reader = csv.reader(input_file)
            csv_writer = csv.writer(output_file)

            for row in csv_reader:
                modified_row = [
                    replace_bingmap(replace_panos(replace_zeros(filename)))
                    for filename in row if "streetview/annotations" not in filename
                ]

                if all(int(filename.split("/")[-1].split(".")[0]) <= 17683 for filename in row):
                    csv_writer.writerow(modified_row)

        print("File names modified and saved to", output_csv_file)
