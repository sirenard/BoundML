import numpy as np

def shifted_geometric_mean(values, shift=1.0):
    values = np.array(values)
    geom_mean = np.exp(np.mean(np.log(values + shift))) - shift
    return geom_mean