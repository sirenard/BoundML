from typing import Callable, Type, Iterable

from pyscipopt import Model, SCIP_RESULT

from boundml.components import Component, BranchingComponent
from boundml.components.components import ComponentList

from inspect import getmro


def first_common_parent(*classes):
    """Find the first common base class for a list of classes."""
    if not classes:
        return object  # Default to base class if empty

    # Get the Method Resolution Order (MRO) for each class
    mros = [getmro(cls) for cls in classes]

    # Iterate through the first MRO (arbitrary choice)
    for candidate in mros[0]:
        # Check if 'candidate' appears in all other MROs before other candidates
        if all(candidate in mro for mro in mros):
            return candidate
    return object  # Fallback

class ConditionalComponent(Component):
    """
    A ConditionalComponent is a Component run one Component among a list depending on their condition

    """
    subtypes = {}
    def __init__(self, *elements: (Component, Callable[[Model], bool])):
        components, conditions = zip(*elements)
        self.components = ComponentList(components)
        self.conditions = conditions
        self.last_observer_index_used = -1

    @classmethod
    def build(cls, subtype: Type[Component]) -> Type[Component]:
        """
        Build dynamically a ConditionalComponent
        Parameters
        ----------
        elements: (Component, Callable[[Model], bool])
            Tuples of Component and its associated condition

        Returns
        -------
        A specific ConditionalComponent object
        """

        name = f"Conditional{subtype.__name__}"

        if name in ConditionalComponent.subtypes:
            t = ConditionalComponent.subtypes[name]
        else:
            t = type(name, (cls, subtype), {})
            ConditionalComponent.subtypes[name] = t
        return t

    def reset(self, model: Model) -> None:
        self.components.reset(model)

    def callback(self, model: Model, passive: bool=True) -> SCIP_RESULT:
        """
        Call the first component with a True condition
        Parameters
        ----------
        model : Model
        passive : bool

        Returns
        -------
        Either the result of the first callback with a true condition, either None if no condition is True
        """
        for i, (condition, component) in enumerate(zip(self.conditions, self.components)):
            if condition(model):
                self.last_observer_index_used = i
                return component.callback(model, passive)

        return None

    def done(self, model: Model) -> None:
        self.components.done(model)

    def get_last_observer_index_used(self):
        return self.last_observer_index_used

    def __str__(self):
        return f"{self.__class__.__name__}({self.components})"

ConditionalBranchingComponent = ConditionalComponent.build(BranchingComponent)