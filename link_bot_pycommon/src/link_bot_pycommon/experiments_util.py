import errno
import json
import os
import pathlib
from datetime import datetime

import git


def experiment_name(nickname, *names):
    nickname = nickname.replace(" ", "-")
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    format_string = "{}_" + "{}_" * (len(names) - 1) + "{}"
    log_path = os.path.join(format_string.format(stamp, sha, *names))
    log_path = os.path.join(nickname, log_path)
    log_path = log_path.replace(" ", "-")
    return log_path


def make_log_dir(full_log_path: pathlib.Path):
    """ https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python """
    if "log_data" not in str(full_log_path):
        raise ValueError("Full log path must contain 'log_data'")
    if not full_log_path.exists():
        full_log_path.mkdir(parents=True)


def write_metadata(metadata, filename, log_path):
    full_log_path = pathlib.Path("log_data") / log_path

    make_log_dir(full_log_path)

    metadata_path = full_log_path / filename
    metadata_file = open(metadata_path, 'w')
    metadata_file.write(json.dumps(metadata, indent=2))
