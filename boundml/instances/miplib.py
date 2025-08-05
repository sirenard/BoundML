import gzip
import os
import shutil
import zipfile
from pathlib import Path
import random
from typing import Callable

import requests
from appdirs import user_cache_dir
from pyscipopt import Model
from tqdm import tqdm

from boundml.instances import Instances
from boundml.solvers import DefaultScipSolver


class MipLibInstances(Instances):
    """
    MipLibInstances allow to iterate through MipLib instances.
    It downloads archive files once and cache it as long with all the extracted instances.
    It is highly recommended to use the argument force_download and force_extract if the process is killed while
    performing one of these operations.
    By default, MipLibInstances iterates through the instances ordered by alphabetical order, a seed can be set to
    randomize the order.
    """
    archives_urls = {
        "collection": "https://miplib.zib.de/downloads/collection.zip",
        "benchmark": "https://miplib.zib.de/downloads/benchmark.zip"
    }

    def __init__(self, subset: str = "benchmark", force_download: bool = False, force_extract: bool = False,
                 filter: Callable[[str], bool] = lambda x: True):
        """
        Parameters
        ----------
        subset : str
            Name of the subset of instances to used, either "collection" or "benchmark"
        force_download : bool
            Force the download even if the file already exists.
        force_extract : bool
            Force the extraction even if the folder already exists.
        filter : Callable[[str], bool]
            Filter instances based on their name (name.mps) to work only with a desired subset of instances
        """
        # Create a cache directory specific to your library
        self.cache_dir = Path(user_cache_dir("boundml"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zip_file = self.cache_dir.joinpath(f"{subset}.zip")
        self.instances_dir = self.cache_dir.joinpath(f"{subset}")

        if subset in MipLibInstances.archives_urls:
            url = MipLibInstances.archives_urls[subset]
        else:
            raise ValueError(f"Unknown subset: {subset}. Must be one of {MipLibInstances.archives_urls}")

        self._download(url, force_download)
        self._extract(force_download or force_extract)

        self.instances_files = [name for name in os.listdir(self.instances_dir) if filter(name)]
        self.instances_files.sort()
        self.index = 0

    def _download(self, url: str, force: bool = False):
        if force or not self.zip_file.exists():
            resp = requests.get(url, stream=True)
            total = int(resp.headers.get('content-length', 0))
            with open(self.zip_file, 'wb') as file, tqdm(
                    desc=str("Downloading MIPLIB instances"),
                    total=total,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)

    def _extract(self, force: bool = False):
        if force or not self.instances_dir.exists():
            if self.instances_dir.exists():
                shutil.rmtree(self.instances_dir)

            print("Extracting collections.zip...")
            self.instances_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.instances_dir)

            files = os.listdir(self.instances_dir)
            with tqdm(
                    desc=str("Extracting instances"),
                    total=len(files),
                    unit='it',
                    unit_scale=True,
                    unit_divisor=1,
            ) as bar:
                for file in files:
                    path = os.path.join(self.instances_dir, file)
                    dest_path = ".".join(path.split(".")[:-1])
                    with gzip.open(path, 'rb') as f_in:
                        with open(dest_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(path)

                    bar.update(1)

    def __next__(self):
        if self.index >= len(self.instances_files):
            raise StopIteration()

        path = os.path.join(self.instances_dir, self.instances_files[self.index])
        model = Model()
        model.setParam("display/verblevel", 0)
        model.readProblem(path)
        self.index += 1
        return model

    def seed(self, seed: int):
        """
        Shuffle the instances based on the given seed
        Parameters
        ----------
        seed : int
        """
        random.Random(seed).shuffle(self.instances_files)
        self.index = 0


if __name__ == "__main__":
    instances = MipLibInstances("benchmark", filter=lambda name: "30n20b8" in name)
    instances.seed(1)

    solver = DefaultScipSolver("relpscost", {"limits/time": 60})
    solver.solve_model(next(instances))
    print(solver["nnodes"], solver["time"], solver["gap"])
