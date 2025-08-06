import os
import random
from typing import Callable

from pyscipopt import Model

from boundml.instances import Instances


class FolderInstances(Instances):
    """
    FolderInstances allows to iterate through the instances in a folder.
    """
    _archives_urls = {
        "collection": "https://miplib.zib.de/downloads/collection.zip",
        "benchmark": "https://miplib.zib.de/downloads/benchmark.zip"
    }

    def __init__(self, folder: str, filter: Callable[[str], bool] = lambda x: True):
        """
        Parameters
        ----------
        folder : str
            Name of the folder containing the instances.
        filter : Callable[[str], bool]
            Filter instances based on their name to work only with a desired subset of instances
        """
        self._instances_dir = folder

        self._instances_files = [name for name in os.listdir(self._instances_dir) if filter(name)]
        self._instances_files.sort()
        self._index = 0

    def __next__(self):
        if self._index >= len(self._instances_files):
            raise StopIteration

        path = os.path.join(self._instances_dir, self._instances_files[self._index])
        model = Model()
        model.setParam("display/verblevel", 0)
        model.readProblem(path)
        self._index += 1
        return model

    def seed(self, seed: int):
        """
        Shuffle the instances based on the given seed
        Parameters
        ----------
        seed : int
        """
        random.Random(seed).shuffle(self._instances_files)
        self._index = 0

    def __len__(self):
        return len(self._instances_files)
