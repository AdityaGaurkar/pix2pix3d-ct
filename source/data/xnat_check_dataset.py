from upload_to_xnat import get_xnat
import progressbar
import sys


if __name__ == "__main__":
    connection = get_xnat()
    project = connection.projects[sys.argv[1]]

    count = {
        "cnn-train": 0,
        "cnn-validation": 0,
        "cnn-test": 0
    }

    experiments = project.experiments.values()
    pb = progressbar.ProgressBar(max_value=len(experiments))
    for idx, experiment in enumerate(experiments):
        pb.update(idx+1)

        for scan in experiment.scans.values():
            if scan.type not in ["sub", "fill", "nat"]:
                if "Sub" in scan.type:
                    scan.type = "sub"
                elif "Fill" in scan.type:
                    scan.type = "fill"
                elif "Mask" in scan.type:
                    scan.type = "nat"

    pb = progressbar.ProgressBar(max_value=len(project.subjects))
    for idx, subject in enumerate(project.subjects.values()):
        pb.update(idx+1)

        usage = None

        for experiment in subject.experiments.values():
            if experiment.fields["cnn_usage"] == "True":
                if usage is None:
                    usage = experiment.fields["usage"]

                assert experiment.fields["usage"] == usage

                count[usage] += 1

        # if this is a validation patient, we can use bad quality 3D-DSA unsuitable
        # for cnn training as additional validation images to compare whether the
        # GAN images are really better than the bad quality original subtraction series
        if usage == "cnn-test":
            for experiment in subject.experiments.values():
                if experiment.fields["cnn_usage"] == "False" and ("usage" not in experiment.fields or experiment.fields["usage"] != "additional-test"):
                    experiment.fields["usage"] = "additional-test"

        # images with unsuitable quality from train or validation cohort must be marked as not used explicitly
        elif usage:
            for experiment in subject.experiments.values():
                if experiment.fields["cnn_usage"] == "False" and ("usage" not in experiment.fields or experiment.fields["usage"] != "none"):
                    experiment.fields["usage"] = "none"

        # images with unsuitable quality whose patient only has one or more such unsuitable quality series can also be
        # used for additional testing
        else:
            for experiment in subject.experiments.values():
                assert experiment.fields["cnn_usage"] == "False"

                if "usage" not in experiment.fields or experiment.fields["usage"] != "additional-test":
                    experiment.fields["usage"] = "additional-test"

    print(count)
    no = sum([c for c in count.values()])
    for k, v in count.items():
        count[k] /= no

    print(count)
    print(sum([c for c in count.values()]))
