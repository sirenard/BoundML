import ecole
import pyscipopt

from boundml.output_control import OutputControl


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


def write_to_file(model_path, solution_path, model: pyscipopt.Model):
    output_control = OutputControl()
    output_control.mute()
    model.writeProblem(model_path, trans=True)
    model.writeBestTransSol(solution_path, write_zeros=True)
    output_control.unmute()

