import errno
import json
import os
from datetime import datetime

import git


def experiment_name(nickname='', additional_attribute=''):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    nickname = "" if nickname is None else nickname.replace(" ", "_")
    if additional_attribute:
        log_path = os.path.join(nickname, "{}__{}__{}".format(stamp, sha, additional_attribute))
    else:
        log_path = os.path.join(nickname, "{}__{}".format(stamp, sha))
    return log_path


def make_log_dir(full_log_path):
    """ https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python """
    if "log_data" not in full_log_path:
        raise ValueError("Full log path must contain 'log_data'")
    try:
        os.makedirs(full_log_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(full_log_path):
            pass
        else:
            raise


def write_metadata(metadata, filename, log_path):
    full_log_path = os.path.join("log_data", log_path)

    make_log_dir(full_log_path)

    metadata_path = os.path.join(full_log_path, filename)
    metadata_file = open(metadata_path, 'w')
    metadata_file.write(json.dumps(metadata, indent=2))
