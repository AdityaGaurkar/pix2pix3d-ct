import os
import sys
import glob
import progressbar

from upload_to_xnat import get_xnat


if __name__ == "__main__":
    connection = get_xnat()
    project = connection.projects[sys.argv[1]]

    # folder containing the PNG-files of those experiment with a suitable image quality
    input_folder = sys.argv[2]

    delete = True if len(sys.argv) > 3 and sys.argv[3] == "true" else False
    print("Delete:", delete)

    if delete is True:
        pb = progressbar.ProgressBar(max_value=len(project.subjects))
        for idx, subject in enumerate(project.subjects.values()):
            pb.update(idx+1)
            if len(subject.experiments) == 0:
                print("Delete empty subject:", subject.label)
                subject.delete()

    experiments = project.experiments.values()
    pb = progressbar.ProgressBar(max_value=len(experiments))

    for idx, experiment in enumerate(experiments):
        pb.update(idx+1)

        # cnn usage is possible if the PNG-folder contains a file with the same session label as a name (+ suffix)
        cnn_usage = True if glob.glob(os.path.join(input_folder, "{}.*".format(experiment.label))) else False

        if delete is True:
            if cnn_usage is not True:
                print("Delete experiment:", experiment.label)
                experiment.delete()

            continue

        # only change the "cnn_usage" field value if it is not yet existing or the value changed
        if "cnn_usage" not in experiment.fields or experiment.fields["cnn_usage"] != str(cnn_usage):
            experiment.fields["cnn_usage"] = str(cnn_usage)

        # if dataset cannot be used for additional validation
        if not cnn_usage:
            if "usage" in experiment.fields:
                print("Delete unsuitable usage", experiment.fields["usage"], "in session", experiment.label)
                del experiment.fields["usage"]

        no_files = None

        if delete is True:
            for scan in experiment.scans.values():
                if not no_files:
                    no_files = len(scan.files)
                elif no_files != len(scan.files):
                    print("Corrupt data:", experiment.label)
                    # experiment.delete()
                    break
