import os
import random
import warnings
from typing import Callable

from pyscipopt import Model

from boundml.core.utils import check_readability
from boundml.instances import Instances


class FolderInstances(Instances):
    """
    FolderInstances allows to iterate through the instances in a folder.
    """

    def __init__(self, folder: str, filter: Callable[[str], bool] = lambda x: True, allow_reading_error=True):
        """
        Parameters
        ----------
        folder : str
            Name of the folder containing the instances.
        filter : Callable[[str], bool]
            Filter instances based on their name to work only with a desired subset of instances
        allow_reading_error: bool
            If True and SCIP failed to read a problem, catch the error and go to the next one.
        """
        self._instances_dir = folder

        self._instances_files = [name for name in os.listdir(self._instances_dir) if filter(name)]
        self._instances_files.sort()
        self._index = 0
        self.allow_reading_error = allow_reading_error

    def __next__(self) -> str:
        if self._index >= len(self._instances_files):
            raise StopIteration

        path = os.path.join(self._instances_dir, self._instances_files[self._index])
        self._index += 1
        model = Model()
        model.setParam("display/verblevel", 0)

        if self.allow_reading_error:
            is_valid = check_readability(path)

            if not is_valid:
                warnings.warn(
                    f"Failed to read problem {path}." + "Skipping to the next one." if self.allow_reading_error else "")
                return next(self)

        return path

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
