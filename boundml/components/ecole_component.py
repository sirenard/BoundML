from pyscipopt import Model

from . import BranchingComponent

try:
    import ecole  # The optional dependency
    HAS_ECOLE_FORK = True
except ImportError:
    HAS_ECOLE_FORK = False

class EcoleComponent(BranchingComponent):
    """
    EcoleComponent is a wrapper around a [ecole](https://github.com/sirenard/ecole) Observer.
    Its callback method returns what the extract method of the Observer would have returned
    """
    def __init__(self, observer):
        super().__init__()

        if not HAS_ECOLE_FORK:
            raise RuntimeError(
                "EcoleComponent requires 'ecole-fork' package. "
                "Install with: pip install boundml[ecole]"
            )
        self.observer = observer
        self.ecole_model = None

    def reset(self, model: Model) -> None:
        self.ecole_model = ecole.scip.Model.from_pyscipopt(model)
        self.observer.before_reset(self.ecole_model)

    def callback(self, model: Model, passive: bool=True):
        self.observation = self.observer.extract(self.ecole_model, done=False)
        return None

    def __getstate__(self):
        return type(self.observer)

    def __setstate__(self, state):
        self.__init__(state())
