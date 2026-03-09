import glob
import os
import shutil

studie_folders = glob.glob("/home/topf/dicom/ai-dsa/storescp/20*/*")

print(len(studie_folders))

for studie_folder in studie_folders:
    sub_folder = os.path.join(studie_folder, "sub")
    fill_folder = os.path.join(studie_folder, "fill")

    assert os.path.exists(sub_folder)
    assert os.path.exists(fill_folder)

    if len(os.listdir(sub_folder)) != len(os.listdir(fill_folder)):
        print(studie_folder, len(os.listdir(sub_folder)), len(os.listdir(fill_folder)))

        # shutil.rmtree(studie_folder)
