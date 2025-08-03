from abc import ABC, abstractmethod
from typing import Iterable

from pyscipopt import Model, SCIP_RESULT



class Component(ABC):
    """
    A Component is a component of a ModularSolver that contains different callback used by the solver.
    Depending on its subtype, it is used at different moment of the solving process
    """

    def __init__(self):
        # Member to store whatever the Component want.
        self.observation = None

    def reset(self, model: Model) -> None:
        """
        Resets the component to its initial state.
        Called by the solver just before starting the solving process.
        Parameters
        ----------
        model : Model
            State of the model
        """
        pass

    @abstractmethod
    def callback(self, model: Model, passive: bool=True) -> SCIP_RESULT:
        """
        Callback method called by the solver.
        Depending on its subtype, it is used at different moment of the solving process

        Parameters
        ----------
        model : Model
            State of the model
        passive : bool
            Whether the component is allowed to perform an action on the model or not

        Returns
        -------
        SCIP_RESULT that corresponds to the action made by the callback, if no action was made then return None
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def done(self, model: Model) -> None:
        """
        Called by the solver once the solving process is done.
        Can be useful to perform final actions.

        Parameters
        ----------
        model : Model
            State of the model
        """
        pass

class ComponentList:
    def __init__(self, components: Iterable[Component]):
        self.components = components

    def __getstate__(self):
        return self.components

    def __setstate__(self, state):
        self.__init__(state)

    def __getattr__(self, name):
        """Forward method calls to all components."""
        def method(*args, **kwargs):
            for component in self.components:
                getattr(component, name)(*args, **kwargs)
        return method

    def __iter__(self):
        return iter(self.components)

    def __str__(self):
        return  ",".join([str(c) for c in self.components])

