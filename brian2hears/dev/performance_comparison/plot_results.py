from matplotlib.pyplot import *

from brian2hears.dev.performance_comparison.benchmark_parameters import parameters
from matplotlib import cm

Ncfs = parameters['Ncfs']
durations = parameters['durations']

databh = np.load(parameters['bh_python_fn'])['arr_0']
datab2h = np.load(parameters['b2h_python_fn'])['arr_0']

subplot(211)
for kncf in range(data.shape[0]):
    plot(durations, datab2h[kncf,:]/databh[kncf,:]*100., c = cm.jet(float(kncf)/(len(Ncfs))))
xlabel('duration')
ylabel('B2H perf / BH perf (%)')


subplot(212)
for kdur in range(data.shape[1]):
    plot(Ncfs, datab2h[:,kdur]/databh[:,kdur]*100., c = cm.jet(float(kdur)/(len(durations))))
xlabel('# of CFs')
ylabel('B2H perf / BH perf (%)')


show()


