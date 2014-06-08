from numpy import *
from matplotlib.pyplot import *
folder = '/Users/victorbenichoux/workspace/lib/brian2hears-git/brian2hears/examples'
t = fromfile(folder+'/anf_cpp/results/_dynamic_array_spikemonitor_t', dtype=float64)
i = fromfile(folder+'/anf_cpp/results/_dynamic_array_spikemonitor_i', dtype=int32)

plot(t, i, '.')
show()
