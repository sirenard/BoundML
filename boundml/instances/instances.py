from abc import abstractmethod, ABC

from pyscipopt import Model


class Instances(ABC):
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Model:
        raise NotImplementedError("Subclasses must implement this method.")