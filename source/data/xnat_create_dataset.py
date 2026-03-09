import sys
import random

from upload_to_xnat import get_xnat
import progressbar


# PLEASE RUN xnat_update_cnn_usage.py BEFOREHAND #

if __name__ == "__main__":
    connection = get_xnat()
    project = connection.projects[sys.argv[1]]

    count = {
        "cnn-train": 0,
        "cnn-validation": 0,
        "cnn-test": 0
    }

    pb = progressbar.ProgressBar(max_value=len(project.subjects))
    for idx, subject in enumerate(project.subjects.values()):
        pb.update(idx+1)
        r = random.random()

        t = "cnn-train" if r < 0.7 else "cnn-validation"
        if r >= 0.9:
            t = "cnn-test"

        count[t] += 1

        for experiment in subject.experiments.values():
            if "cnn_usage" in experiment.fields and experiment.fields["cnn_usage"] == "True":
                experiment.fields["usage"] = t

    print(count)
    no = sum([c for c in count.values()])
    for k, v in count.items():
        count[k] /= no

    print(count)
    print(sum([c for c in count.values()]))
