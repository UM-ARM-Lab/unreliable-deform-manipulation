import os
from datetime import datetime
import git


def experiment_name(nickname=None):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    stamp = "{:%B_%d_%H:%M:%S}".format(datetime.now())
    nickname = "" if nickname is None else nickname.replace(" ", "_")
    log_path = os.path.join(nickname, "{}__{}".format(stamp, sha))
    return log_path
