from brian2 import *
from brian2hears import *

duration = 10*second
itd = 100*us

n_itd = np.rint(itd/defaultclock.dt)

print n_itd

cfs = erbspace(100*Hz, 2000*Hz, 16)
sound_left = TimedArray(np.random.randn(np.rint(duration/defaultclock.dt)), dt = defaultclock.dt)
sound_right = TimedArray(np.hstack((np.zeros(n_itd), sound_left.values)), dt = defaultclock.dt)

left_cochlea = GammatoneFilterbank(sound_left, cfs)
right_cochlea = GammatoneFilterbank(sound_right, cfs)

# Leaky integrate-and-fire model with noise and refractoriness
eqs = '''
dv/dt = (I_in-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1
I_in = 3*clip(I, 0, Inf)**(1./3.) : 1
'''
anf_left = NeuronGroup(len(cfs), eqs, reset='v=0', threshold='v>1')#, refractory=5*ms)
anf_left.variables.add_reference('I', right_cochlea.variables['out'])

anf_right = NeuronGroup(len(cfs), eqs, reset='v=0', threshold='v>1')#, refractory=5*ms)
anf_right.variables.add_reference('I', right_cochlea.variables['out'])

delays = np.linspace(0*us, 500*us, 10)
binaural_cells = NeuronGroup(len(cfs)*len(delays), '''
dv/dt = -v/(0.5*ms) : 1
''', threshold = 'v>1', reset = 'v=1')

Sl = Synapses(anf_left, binaural_cells, model = 'w:1', pre = 'v += w')
Sr = Synapses(anf_left, binaural_cells, model = 'w:1', pre = 'v += w')
for kcf in range(len(cfs)):
    Sr.connect(kcf, np.arange(len(delays)))
    Sl.connect(kcf, np.arange(len(delays)))
    

M = SpikeMonitor(anf_left)

net = Network(anf_left, anf_right, Sl, Sr, binaural_cells, M)
net.run(len(sound_left.values)*defaultclock.dt)

i, t = M.it

plot(np.array(t), i, '.', color = 'k')

show()



