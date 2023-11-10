import os
import shutil

# Define the root folder where all the numbered folders are located.
# root_folders = ["C:\\Users\\lordres\\Desktop\\Dyplom\\University-Release\\Test\\query_images",
#                 "C:\\Users\\lordres\\Desktop\\Dyplom\\University-Release\\Test\\reference_images",
#                 "C:\\Users\\lordres\\Desktop\\Dyplom\\University-Release\\Train\\query_images",
#                 "C:\\Users\\lordres\\Desktop\\Dyplom\\University-Release\\Train\\reference_images"]

root_folders = ["C:\\Users\\lordres\\Desktop\\Dyplom\\University-Release\\Test\\reference_images",
                "C:\\Users\\lordres\\Desktop\\Dyplom\\University-Release\\Train\\reference_images"]

# # Loop through each numbered folder
# for root_folder in root_folders:
#     for folder_name in os.listdir(root_folder):
#         print(folder_name)
#         folder_path = os.path.join(root_folder, folder_name)
#
#         # Check if it's a directory
#         if os.path.isdir(folder_path):
#             # Extract the numeric part of the folder name
#             index = folder_name.lstrip('0')
#
#             # Pad the index with zeros to match the '01' format
#             index = index.zfill(2)
#
#             # Rename image-01.jpeg to image-{index}.jpeg in the subfolder
#             old_image_path = os.path.join(folder_path, 'image-01.jpeg')
#             new_image_path = os.path.join(folder_path, f'image-00{index}.jpeg')
#             os.rename(old_image_path, new_image_path)
#
#             # Move the renamed image to the root folder
#             shutil.move(new_image_path, os.path.join(root_folder, os.path.basename(new_image_path)))
for root_folder in root_folders:
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # Check if the item in the root folder is a directory (folder)
        if os.path.isdir(folder_path):
            # Loop through each file in the numbered folder
            for filename in os.listdir(folder_path):
                src_path = os.path.join(folder_path, filename)
                dst_path = os.path.join(root_folder, filename)

                shutil.move(src_path, dst_path)

            # Delete the numbered folder
            os.rmdir(folder_path)






