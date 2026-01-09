from abc import abstractmethod, ABC

from pyscipopt import Model


class Instances(ABC):
    """
    An Instances object is an iterator that yields pyscipopt.Model or path to an instance.
    """
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Model | str :
        raise NotImplementedError("Subclasses must implement this method.")