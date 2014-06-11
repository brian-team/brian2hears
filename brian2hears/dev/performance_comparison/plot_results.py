from matplotlib.pyplot import *

from brian2hears.dev.performance_comparison.benchmark_parameters import parameters
from matplotlib import cm

Ncfs = parameters['Ncfs']
durations = parameters['durations']
fn = parameters['b2h_python_fn']

results_fns = [parameters['b2h_python_fn'], parameters['bh_python_fn']]

for kfn in range(len(results_fns)):
    data = np.load(results_fns[kfn])['arr_0']
    for kncf in range(data.shape[0]):
        plot(durations, data[kncf,:])#, c = cm.jet(float(kncf)/(Ncfs.max()))


show()


