from pyscipopt import Model
from .components import Component

try:
    import ecole  # The optional dependency
    HAS_ECOLE_FORK = True
except ImportError:
    HAS_ECOLE_FORK = False

class EcoleComponent(Component):
    """
    EcoleComponent is a wrapper around a [ecole](https://github.com/sirenard/ecole) Observer.
    Its callback method returns what the extract method of the Observer would have returned
    """
    def __init__(self, observer):
        if not HAS_ECOLE_FORK:
            raise RuntimeError(
                "EcoleComponent requires 'ecole-fork' package. "
                "Install with: pip install boundml[ecole]"
            )
        self.observer = observer
        self.ecole_model = None
        self.observation = None

    def reset(self, model: Model) -> None:
        self.ecole_model = ecole.scip.Model.from_pyscipopt(model)
        self.observer.reset(self.ecole_model)

    def callback(self, model: Model, passive: bool=True):
        self.observation = self.observer.extract(self.ecole_model, done=False)
        return None