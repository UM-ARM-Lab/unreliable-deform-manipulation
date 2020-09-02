from typing import Optional


def mkdir_and_ask(path, parents: bool, yes: Optional[bool] = False):
    if path.exists():
        msg = f"Path {path} exists, do you want to reuse it? [Y/n]"
        if yes:
            print(f"{msg} answering yes")
            return True
        else:
            response = input(msg)
            if response == 'y' or response == 'Y' or response == '':
                return True
            else:
                return False

    path.mkdir(parents=parents, exist_ok=False)
    return True
