import ecole
import numpy as np
import pyscipopt

def get_ecole_model_from_file(path):
    model = pyscipopt.Model()
    model.readProblem(path)
    return ecole.scip.Model.from_file(path)


def configure(model: pyscipopt.Model, branching_policy="vanillafullstrong"):
    model.setParam("randomization/permutevars", False)
    model.setParam("randomization/lpseed", 1)
    model.setParam("randomization/randomseedshift", 1)
    model.setParam("randomization/permutationseed", 1)
    model.setParam("display/verblevel", 0)

    model.setParam(f"branching/{branching_policy}/priority", 9999999)


def solve(model: pyscipopt.Model):
    model.optimize()
    nnodes = model.getNNodes()
    obj = model.getObjVal()
    return nnodes, obj

def shifted_geometric_mean(values, shift=1.0):
    values = np.array(values)
    geom_mean = np.exp(np.mean(np.log(values + shift))) - shift
    return geom_mean