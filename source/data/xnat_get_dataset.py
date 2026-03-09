import sys

from upload_to_xnat import get_xnat
import os
import shutil
import glob
import progressbar


if __name__ == "__main__":
    usages = [
        "cnn-train",
        "cnn-validation",
        "cnn-test",
        "additional-test",
    ]

    scan_types = ["sub", "fill"]

    dataset_folder = sys.argv[2]

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    for t in usages:
        t_folder = os.path.join(dataset_folder, t)
        if not os.path.exists(t_folder):
            os.makedirs(t_folder)

    for folder in os.listdir(dataset_folder):
        if folder not in usages:
            print("Remove folder:", folder)
            shutil.rmtree(os.path.join(dataset_folder, folder))

    connection = get_xnat()
    project = connection.projects[sys.argv[1]]

    print("Creating (empty) series folders")
    experiments = project.experiments.values()
    pb = progressbar.ProgressBar(max_value=len(experiments))
    for idx, experiment in enumerate(experiments):
        pb.update(idx+1)
        if "usage" in experiment.fields and experiment.fields["usage"] != "none":
            series_folder = os.path.join(dataset_folder, experiment.fields["usage"], experiment.label)

            if not os.path.exists(series_folder):
                os.makedirs(series_folder)

    # clean wrong series
    series_folders = glob.glob(os.path.join(dataset_folder, "*", "*"))
    print("Currently, there exist {} series in the dataset folder".format(len(series_folders)))

    pb = progressbar.ProgressBar(max_value=len(series_folders))
    for idx, series_folder in enumerate(series_folders):
        pb.update(idx+1)
        series_label = os.path.basename(series_folder)
        series_type = os.path.basename(os.path.dirname(series_folder))
        # print(series_folder, series_label, series_type)

        if series_label not in connection.experiments:
            print("Remove unknown series {} -> {}".format(series_type, series_label))
            shutil.rmtree(series_folder)
            continue

        experiment = connection.experiments[series_label]
        if series_type != experiment.fields["usage"]:
            print("Wrong series type ({} instead of {}) for series {}, delete".format(series_type, experiment.fields["usage"], series_label))
            shutil.rmtree(series_folder)
            continue

        for t_folder in os.listdir(series_folder):
            scan_folder = os.path.join(series_folder, t_folder)
            if t_folder not in scan_types:
                print("Delete unknown scan type {} for series {}".format(t_folder, series_label))
                shutil.rmtree(scan_folder)
                continue

        for scan_type in scan_types:
            scan_folder = os.path.join(series_folder, scan_type)

            if not os.path.exists(scan_folder):
                os.makedirs(scan_folder)

            xnat_scan = experiment.scans[scan_type]

            xnat_scan_files = {x.name: x for x in xnat_scan.files.values() if x.name.endswith(".dcm")}

            for wrong_dicom_file in [os.path.join(scan_folder, d) for d in os.listdir(scan_folder) if d not in xnat_scan_files]:
                print("Delete wrong file/folder {}".format(wrong_dicom_file))
                if os.path.isdir(wrong_dicom_file):
                    shutil.rmtree(wrong_dicom_file)
                else:
                    os.remove(wrong_dicom_file)

            for xnat_scan_file in xnat_scan_files.values():
                dicom_file = os.path.join(scan_folder, xnat_scan_file.name)
                if not os.path.exists(dicom_file):
                    xnat_scan_file.download(dicom_file, verbose=False)
