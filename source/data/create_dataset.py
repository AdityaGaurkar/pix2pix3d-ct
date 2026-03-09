import glob
import os
import sys
import shutil
import random


if __name__ == "__main__":
    # something like "<..>/20*/*"
    input_folders = glob.glob(sys.argv[1])

    print("Input folders found:", len(input_folders))

    # something like "<..>/dataset"
    output_folder = sys.argv[2]

    # folder containing png images montages of suitable images
    suitable_series_folder = sys.argv[3]

    suitable_series = {f[:-4]: None for f in os.listdir(suitable_series_folder)}

    print("{} suitable DSA series selected".format(len(suitable_series)))

    for series_id in suitable_series.keys():
        series_folders = [f for f in input_folders if os.path.basename(f) == series_id]

        assert len(series_folders) == 1

        suitable_series[series_id] = series_folders[0]

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    folder_train = os.path.join(output_folder, "train")
    folder_test = os.path.join(output_folder, "test")
    folder_validation = os.path.join(output_folder, "validation")

    os.makedirs(folder_train)
    os.makedirs(folder_test)
    os.makedirs(folder_validation)

    folders = list(suitable_series.values())

    random.shuffle(folders)
    train_data = folders[:int((len(folders) + 1) * .70)]  # Remaining 70% to training set
    test_data = folders[int((len(folders) + 1) * .70):int((len(folders) + 1) * .90)]  # Splits 20% data to test set
    validation_data = folders[int((len(folders) + 1) * .90):]  # Splits 20% data to test set

    for d in train_data:
        # os.makedirs(os.path.join(folder_train, os.path.basename(d)))
        os.symlink(d, os.path.join(folder_train, os.path.basename(d)))

    for d in test_data:
        # os.makedirs(os.path.join(folder_test, os.path.basename(d)))
        os.symlink(d, os.path.join(folder_test, os.path.basename(d)))

    for d in validation_data:
        # os.makedirs(os.path.join(folder_validation, os.path.basename(d)))
        os.symlink(d, os.path.join(folder_validation, os.path.basename(d)))
