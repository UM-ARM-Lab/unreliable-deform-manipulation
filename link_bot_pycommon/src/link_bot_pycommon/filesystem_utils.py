def mkdir_and_ask(path, parents: bool):
    if path.exists():
        response = input("Path {} exists, do you want to reuse it? [Y/n]".format(path))
        if response == 'y' or response == 'Y' or response == '':
            return True
        else:
            return False

    path.mkdir(parents=parents, exist_ok=False)
    return True
