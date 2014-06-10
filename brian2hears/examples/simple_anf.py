#!/usr/bin/env python
'''
Example of a simple auditory nerve fibre model with Brian hears.
'''
from brian2 import *
from brian2hears import *

N = 1000
Ntaps = 20.

sound = TimedArray(np.random.randn(N), dt = defaultclock.dt)

cf = np.array(erbspace(20*Hz, 20000.*Hz, 3000))
cochlea = GammatoneFilterbank(sound, cf)

# Leaky integrate-and-fire model with noise and refractoriness
eqs = '''
dv/dt = (I_in-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1
I_in = 3*clip(I, 0, Inf)**(1./3.) : 1
'''
anf = NeuronGroup(len(cf), eqs, reset='v=0', threshold='v>1')#, refractory=5*ms)
anf.variables.add_reference('I', cochlea.variables['out'])

M = SpikeMonitor(anf)
run(len(sound.values)*defaultclock.dt)
i, t = M.it
plot(np.array(t), i, '.', color = 'k')

show()
