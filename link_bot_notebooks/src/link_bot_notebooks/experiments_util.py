import errno
import os
from datetime import datetime
import git


def experiment_name(nickname='', additional_attribute=''):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    nickname = "" if nickname is None else nickname.replace(" ", "_")
    log_path = os.path.join(nickname, "{}__{}__{}".format(stamp, sha, additional_attribute))
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
