from typing import Any

from pyscipopt import Model


class Observer:
    """
    An Observer is an object that is called before each branching decisions to make an observation of the solver state.
    """
    def __init__(self, seed: int = None, principal_observer: bool = False):
        """
        Parameters
        ----------
        seed : int
            Seed of the observer
        principal_observer : bool
            Whether the observer is the main observer, i.e., the one used that compute the scores of the candidates.
        """
        self.seed = seed
        self.instance_path = None
        self.principal_observer = principal_observer

    def before_reset(self, model: Model):
        """
        Callback method called before the reset of the environment.
        Can be used to extract information from the model before it is reset

        Parameters
        ----------
        model : Model
            State of the model
        """
        return

    def extract(self, model: Model, done: bool) -> Any:
        """
        Callback method called before each branching decision.
        Can be used to extract any information from the model.

        Parameters
        ----------
        model :
            State of the model
        done : bool
            Whether the solving process is done

        Returns
        -------
        Anything the user want.
        If the Observer is used as a scoring observer that scores each candidate, it must return a numpy array of size
        model.getNVars() with float value for branching candidates variables that are their score, and np.nan values for
        the other variables.
        """
        return None

    def reset(self, instance_path, seed=None):
        """
        Callback method called after the reset of the environment.
        Can be used to reset attribute.

        Parameters
        ----------
        instance_path : str
            Path to the instance that will be solved by the solver
        seed : int
        """
        self.instance_path = instance_path
        self.seed = seed

    def done(self, model: Model):
        """
        Callback method once the solving process is done.

        Parameters
        ----------
        model : Model
            State of the model
        """
        return

    def set_principal_observer(self, val):
        self.principal_observer = val

    def is_principal_observer(self):
        return self.principal_observer

    def __str__(self):
        return "default"
