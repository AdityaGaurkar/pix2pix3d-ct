import glob
import sys

import xnat.exceptions

import os
import shutil
import tempfile
import multiprocessing as mp


def get_xnat():
    return xnat.connect("https://xnat1.diz-ag.med.ovgu.de", os.getenv("XNAT_USER"), password=os.getenv("XNAT_PW"), )


def zip(_queue: mp.Queue, _queue_upload: mp.Queue):
    while True:
        _series_dir = _queue.get()
        if isinstance(_series_dir, list):
            _series_dir, _patient_id, _session_label = _series_dir
        else:
            _patient_id = None,
            _session_label = None

        print(_queue.qsize(), os.path.basename(_series_dir))

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=os.environ.get("TEMP_FOLDER") if "TEMP_FOLDER" in os.environ else None) as zip_file:
            shutil.make_archive(zip_file.name[:-4], "zip", _series_dir)
            _queue_upload.put((zip_file.name, os.path.basename(_series_dir), _patient_id, _session_label, 0))


def upload(_connection, _queue: mp.Queue, _queue_upload: mp.Queue, _project: str):
    while True:
        temp_file, series_dir, patient_id, session_label, retry = _queue_upload.get()
        try:
            # print(temp_file, series_dir)
            if not _connection:
                _connection = get_xnat()

            if isinstance(patient_id, str):
                _connection.services.import_(temp_file, project=_project, destination="prearchive",
                                        overwrite="append", subject=patient_id, experiment=session_label)
            else:
                _connection.services.import_(temp_file, project=_project, destination="prearchive",
                                             overwrite="append", experiment=session_label)

            os.remove(temp_file)

        except xnat.exceptions.XNATUploadError as err:
            print(err)
            os.remove(temp_file)

        except Exception as err:
            if retry > 20:
                print("Stop trying to upload {} after {} retries".format(series_dir, retry))
                os.remove(temp_file)
            else:
                _queue_upload.put((temp_file, series_dir, patient_id, session_label, retry + 1))

            _connection = get_xnat()


def get_queue(project: str, queue_size: int = 0, upload_queue_size: int = 1, processes_zip: int = 2, processes_upload: int = 3):
    queue = mp.Queue(maxsize=queue_size)
    queue_upload = mp.Queue(maxsize=upload_queue_size)

    for i in range(processes_zip):
        connection = get_xnat()
        p = mp.Process(target=upload, args=((connection), (queue), (queue_upload), (project)))
        p.start()

    for i in range(processes_upload):
        p = mp.Process(target=zip, args=((queue), (queue_upload)))
        p.start()

    return queue


if __name__ == "__main__":
    project = sys.argv[1]
    experiment_dirs = glob.glob(sys.argv[2])

    connection = get_xnat()
    subjects = connection.projects[project]

    experiments = connection.projects[project].experiments

    series_dirs = []

    prearchive_sessions = connection.prearchive.sessions()

    for experiment_dir in experiment_dirs:
        experiment_id = os.path.basename(experiment_dir)

        # already uploaded and archived?
        if experiment_id in experiments:
            continue

        # already uploaded but still in prearchiv?
        in_prearchiv = False
        for prearchive_session in prearchive_sessions:
            if prearchive_session.name == experiment_id:
                in_prearchiv = True
                break

        if not in_prearchiv:
            series_dirs.append(experiment_dir)

    print(len(series_dirs), "/", len(experiment_dirs))

    queue = mp.Queue(maxsize=len(series_dirs))
    queue_upload = mp.Queue(maxsize=3)

    for series_dir in series_dirs:
        queue.put(series_dir)

    for i in range(8):
        connection = get_xnat()
        p = mp.Process(target=upload, args=((connection), (queue), (queue_upload), (project)))
        p.start()

    for i in range(4):
        p = mp.Process(target=zip, args=((queue), (queue_upload)))
        p.start()
