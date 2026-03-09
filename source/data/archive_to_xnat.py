import sys
import threading
from queue import Queue

import xnat

from upload_to_xnat import get_xnat


def archive(queue: Queue):
    while not queue.empty():
        print(queue.qsize(), "/", queue.maxsize)
        session = queue.get()
        res = None
        try:
            res = session.archive(overwrite="append")
        except xnat.exceptions.XNATResponseError as err:
            print(err, res)


def show_prearchiv(connection, project_id: str = None):
    # print(len(connection.prearchive.sessions()), len([s for s in connection.prearchive.sessions() if s.label == s.subject]))

    # sessions = [session for session in connection.prearchive.sessions() if session.project == "TICI-SCORING"]
    sessions = connection.prearchive.sessions()

    queue = Queue(maxsize=len(sessions))
    for session in sessions:
        if session.project == project_id:
            queue.put(session)

    threads = []
    no_threads = 6
    for i in range(no_threads):
        thread = threading.Thread(target=archive, args=(queue,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    project = sys.argv[1]

    connection = get_xnat()
    show_prearchiv(connection, project_id=project)
