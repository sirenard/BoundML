import sys
import types
from enum import Enum
import numpy as np


def setup_fake_environment():
    """Install lightweight stand-ins for optional dependencies used by the library."""
    _ensure_pathos_stub()
    _ensure_pyscipopt_stub()
    _ensure_boundml_ml_stub()
    _ensure_matplotlib_stub()
    _ensure_dill_stub()
    _ensure_pandas_stub()
    _ensure_tabulate_stub()
    _ensure_appdirs_stub()
    _ensure_requests_stub()
    _ensure_tqdm_stub()


def _ensure_pathos_stub():
    if "pathos.multiprocessing" in sys.modules:
        return

    mp_module = types.ModuleType("pathos.multiprocessing")

    def cpu_count():
        return 1

    class _AsyncResult:
        def __init__(self, func, args):
            self._func = func
            self._args = args

        def get(self):
            return self._func(*self._args)

    class Pool:
        def __init__(self, *_, **__):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def apply_async(self, func, args):
            return _AsyncResult(func, args)

    mp_module.cpu_count = cpu_count
    mp_module.Pool = Pool

    pathos_module = types.ModuleType("pathos")
    pathos_module.multiprocessing = mp_module

    sys.modules["pathos"] = pathos_module
    sys.modules["pathos.multiprocessing"] = mp_module


def _ensure_pyscipopt_stub():
    if "pyscipopt" in sys.modules:
        return

    module = types.ModuleType("pyscipopt")

    class SCIP_RESULT(Enum):
        DIDNOTRUN = 0
        BRANCHED = 1
        CUTOFF = 2

    class Branchrule:
        pass

    class Column:
        def __init__(self, lp_pos):
            self._lp_pos = lp_pos

        def getLPPos(self):
            return self._lp_pos

    class Variable:
        def __init__(self, name, obj=0.0, lpsol=0.0, lp_pos=0):
            self.name = name
            self._obj = obj
            self._lpsol = lpsol
            self._column = Column(lp_pos)

        def getObj(self):
            return self._obj

        def getLPSol(self):
            return self._lpsol

        def getCol(self):
            return self._column

    class Model:
        def __init__(self):
            self.params = {}
            self.branch_rules = []
            self._lp_cands = ([], [], [], 0, 0, 0)
            self.pseudocosts = {}
            self.strongbranch_data = {}
            self.branch_calls = []
            self.branch_scores = []
            self._time = 0.0
            self._nnodes = 0
            self._obj = 0.0
            self._gap = 0.0
            self._best_sol = None
            self._tree_est = 0.0
            self._depth = 0

        # ---- configuration -------------------------------------------------
        def setParam(self, name, value):
            self.params[name] = value

        def setParams(self, params):
            self.params.update(params)

        def includeBranchrule(self, branchrule, name, desc, priority, maxdepth, maxbounddist):
            self.branch_rules.append(
                {
                    "branchrule": branchrule,
                    "name": name,
                    "desc": desc,
                    "priority": priority,
                    "maxdepth": maxdepth,
                    "maxbounddist": maxbounddist,
                }
            )

        def build_dummy_statistics(self, *, time=0.0, nnodes=0, obj=0.0, gap=0.0, tree_est=0.0, best_sol=None):
            self._time = time
            self._nnodes = nnodes
            self._obj = obj
            self._gap = gap
            self._tree_est = tree_est
            self._best_sol = best_sol

        # ---- solving pipeline ----------------------------------------------
        def readProblem(self, filename):
            self._last_problem = filename

        def writeProblem(self, filename, verbose=False):
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write("\\ LP stub\\n")

        def optimize(self):
            for info in self.branch_rules:
                branchrule = info["branchrule"]
                if hasattr(branchrule, "branchexeclp"):
                    branchrule.branchexeclp(False)

        def set_lp_branch_cands(self, candidates, sols=None, fracs=None, counts=None):
            sols = sols if sols is not None else [0.0] * len(candidates)
            fracs = fracs if fracs is not None else [0.0] * len(candidates)
            if counts is None:
                counts = (len(candidates), len(candidates), 0)
            self._lp_cands = (candidates, sols, fracs, *counts)

        def getLPBranchCands(self):
            return self._lp_cands

        def getDepth(self):
            return self._depth

        def setDepth(self, depth):
            self._depth = depth

        def infinity(self):
            return 1e20

        def branchVarVal(self, variable, value):
            self.branch_calls.append((variable, value))

        # ---- pseudocosts & strong branching --------------------------------
        def set_pseudocost(self, variable, score):
            self.pseudocosts[variable] = score

        def getVarPseudocostScore(self, variable, lpsol):
            return self.pseudocosts.get(variable, 0.0)

        def startStrongbranch(self):
            self.strongbranch_started = True

        def set_strongbranch_data(self, variable, data):
            self.strongbranch_data[variable] = data

        def getVarStrongbranch(self, variable, *_, **__):
            return self.strongbranch_data[variable]

        def getLPObjVal(self):
            return getattr(self, "_lp_obj_value", 0.0)

        def setLPObjVal(self, value):
            self._lp_obj_value = value

        def getBranchScoreMultiple(self, variable, gains):
            self.branch_scores.append((variable, gains))
            return sum(gains)

        def endStrongbranch(self):
            self.strongbranch_started = False

        # ---- stats getters --------------------------------------------------
        def getSolvingTime(self):
            return self._time

        def getNTotalNodes(self):
            return self._nnodes

        def getObjVal(self):
            return self._obj

        def getGap(self):
            return self._gap

        def getBestSol(self):
            return self._best_sol

        def getTreesizeEstimation(self):
            return self._tree_est

    module.Model = Model
    module.SCIP_RESULT = SCIP_RESULT
    module.Branchrule = Branchrule
    module.Variable = Variable

    sys.modules["pyscipopt"] = module


def _ensure_boundml_ml_stub():
    if "boundml.ml" in sys.modules:
        return

    ml_module = types.ModuleType("boundml.ml")
    ml_module.__path__ = []

    model_module = types.ModuleType("boundml.ml.model")

    def get_device(try_use_gpu=False):
        return "cpu"

    class DummyTensor:
        def __init__(self, values):
            self._values = np.array(values, dtype=float)

        def cpu(self):
            return self

        def detach(self):
            return self

        def numpy(self):
            return self._values

    class DummyPolicy:
        def __call__(self, *_args):
            # Always return a tensor with values increasing with index.
            var_count = _args[4]
            return DummyTensor(np.arange(var_count, dtype=float))

    def load_policy(*_args, **_kwargs):
        return DummyPolicy()

    model_module.get_device = get_device
    model_module.load_policy = load_policy

    ml_module.model = model_module
    ml_module.load_policy = load_policy
    ml_module.get_device = get_device

    sys.modules["boundml.ml"] = ml_module
    sys.modules["boundml.ml.model"] = model_module


def _ensure_matplotlib_stub():
    if "matplotlib" in sys.modules:
        return

    matplotlib_module = types.ModuleType("matplotlib")
    matplotlib_module._backend = "agg"

    def use(name):
        matplotlib_module._backend = name

    def get_backend():
        return matplotlib_module._backend

    matplotlib_module.use = use
    matplotlib_module.get_backend = get_backend

    pyplot_module = types.ModuleType("matplotlib.pyplot")

    def _noop(*_, **__):
        return None

    pyplot_module.plot = _noop
    pyplot_module.legend = _noop
    pyplot_module.xlabel = _noop
    pyplot_module.ylabel = _noop
    pyplot_module.title = _noop
    pyplot_module.xscale = _noop
    pyplot_module.show = _noop
    pyplot_module.savefig = _noop

    matplotlib_module.pyplot = pyplot_module

    sys.modules["matplotlib"] = matplotlib_module
    sys.modules["matplotlib.pyplot"] = pyplot_module


def _ensure_dill_stub():
    if "dill" in sys.modules:
        return

    dill_module = types.ModuleType("dill")

    def dump(*_args, **_kwargs):
        raise RuntimeError("dill.dump is not implemented in stub")

    def load(*_args, **_kwargs):
        raise RuntimeError("dill.load is not implemented in stub")

    dill_module.dump = dump
    dill_module.load = load

    sys.modules["dill"] = dill_module


def _ensure_pandas_stub():
    if "pandas" in sys.modules:
        return

    pandas_module = types.ModuleType("pandas")

    class ColumnIndex(list):
        def get_level_values(self, level):
            res = []
            for item in self:
                if isinstance(item, tuple):
                    actual_level = level
                    if level < 0:
                        actual_level = len(item) + level
                    res.append(item[actual_level])
                else:
                    res.append(item)
            return ColumnIndex(res)

    class DataFrame:
        def __init__(self, data):
            self._data = dict(data)
            self.columns = ColumnIndex(list(self._data.keys()))
            self._index_key = None
            self._index_values = None

        def set_index(self, key, inplace=True):
            values = self._data.pop(key, None)
            self.columns = ColumnIndex([col for col in self.columns if col != key])
            self._index_key = key
            self._index_values = values
            if not inplace:
                return self

        def to_dict(self, orient="dict"):
            if orient != "list":
                return dict(self._data)
            return {key: list(value) for key, value in self._data.items()}

        def reset_index(self):
            data = dict(self._data)
            if self._index_values is not None:
                data["index"] = list(self._index_values)
            return DataFrame(data)

        def rename(self, columns=None):
            if columns is None:
                return self
            data = {}
            for key, value in self._data.items():
                new_key = columns.get(key, key)
                data[new_key] = value
            return DataFrame(data)

    def DataFrameFactory(data):
        return DataFrame(data)

    def concat(frames, axis=0):
        if axis == 1:
            data = {}
            for frame in frames:
                data.update(frame._data)
            return DataFrame(data)
        raise NotImplementedError("Only axis=1 concatenation is supported in stub.")

    pandas_module.DataFrame = DataFrameFactory
    pandas_module.concat = concat

    sys.modules["pandas"] = pandas_module


def _ensure_tabulate_stub():
    if "tabulate" in sys.modules:
        return

    tabulate_module = types.ModuleType("tabulate")

    def tabulate(dataframe, headers="keys", tablefmt=None, showindex=None):
        columns = list(dataframe._data.keys())
        lines = [" | ".join(str(col) for col in columns)]
        rows = zip(*[dataframe._data[col] for col in columns]) if columns else []
        for row in rows:
            lines.append(" | ".join(str(value) for value in row))
        return "\n".join(lines)

    tabulate_module.tabulate = tabulate

    sys.modules["tabulate"] = tabulate_module


def _ensure_appdirs_stub():
    if "appdirs" in sys.modules:
        return

    appdirs_module = types.ModuleType("appdirs")

    def user_cache_dir(appname):
        return f"/tmp/{appname}"

    appdirs_module.user_cache_dir = user_cache_dir
    sys.modules["appdirs"] = appdirs_module


def _ensure_requests_stub():
    if "requests" in sys.modules:
        return

    requests_module = types.ModuleType("requests")

    class Response:
        def __init__(self):
            self.headers = {"content-length": "0"}

        def iter_content(self, chunk_size=1024):
            return []

    def get(*_args, **_kwargs):
        return Response()

    requests_module.get = get

    sys.modules["requests"] = requests_module


def _ensure_tqdm_stub():
    if "tqdm" in sys.modules:
        return

    tqdm_module = types.ModuleType("tqdm")

    class tqdm:
        def __init__(self, *_, **__):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, *_):
            pass

    tqdm_module.tqdm = tqdm

    sys.modules["tqdm"] = tqdm_module
