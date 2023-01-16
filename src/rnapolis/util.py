import gzip
import os
import tempfile
from typing import IO


def handle_input_file(path) -> IO[str]:
    root, ext = os.path.splitext(path)

    if ext == ".gz":
        root, ext = os.path.splitext(root)
        file = tempfile.NamedTemporaryFile("wt+", suffix=ext)
        with gzip.open(path, "rt") as f:
            file.write(f.read())
            file.seek(0)
    else:
        file = tempfile.NamedTemporaryFile("wt+", suffix=ext)
        with open(path) as f:
            file.write(f.read())
            file.seek(0)
    return file
