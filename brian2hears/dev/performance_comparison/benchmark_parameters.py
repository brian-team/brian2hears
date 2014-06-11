import numpy as np


parameters = {
    'Ncfs' : np.array([10, 100, 1000, 10000]),
    'durations' : np.array([0.1, 1., 10., 100.]),
    'bh_python_fn' : './brianhears_python_results.npz',
    'b2h_python_fn' : './brian2hears_python_results.npz'
}

