from pyscipopt import Model

from boundml.instances.instances import Instances

try:
    import ecole  # The optional dependency
    HAS_ECOLE_FORK = True
except ImportError:
    HAS_ECOLE_FORK = False

class EcoleInstances(Instances):
    def __init__(self, ecole_instances):
        self.ecole_instances = ecole_instances

    def __next__(self) -> Model:
        ecole_instance: ecole.scip.Model = next(self.ecole_instances)
        return ecole_instance.as_pyscipopt()

    def seed(self, seed):
        self.ecole_instances.seed(seed)

class CapacitatedFacilityLocationGenerator(EcoleInstances):
    def __init__(self, *args, **kwargs):
        super().__init__(ecole.instance.CapacitatedFacilityLocationGenerator(*args, **kwargs))

class CombinatorialAuctionGenerator(EcoleInstances):
    def __init__(self, *args, **kwargs):
        super().__init__(ecole.instance.CombinatorialAuctionGenerator(*args, **kwargs))

class IndependentSetGenerator(EcoleInstances):
    def __init__(self, *args, **kwargs):
        super().__init__(ecole.instance.IndependentSetGenerator(*args, **kwargs))

class SetCoverGenerator(EcoleInstances):
    def __init__(self, *args, **kwargs):
        super().__init__(ecole.instance.SetCoverGenerator(*args, **kwargs))
