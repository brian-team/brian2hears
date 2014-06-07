from brian2 import *
from brian2hears.library.gammatone import *
import numpy as np
import time
#brian_prefs.codegen.target = 'weave'

f = open('./temp.npz', 'r')
tmp = np.load(f)
signal = tmp['signal']
cf = tmp['cf']
bh_out = tmp['bh_out']
f.close()
samplerate = 1./defaultclock.dt
sound = TimedArray(signal, dt = defaultclock.dt)

iir_filter = GammatoneFilterbank(sound, cf)
Miir = StateMonitor(iir_filter, 
'out', record = True)
iir_net = Network(iir_filter, Miir)
t0 = time.time()
iir_net.run(len(signal)*defaultclock.dt)
print 'IIR in ', time.time()-t0
iir_out = Miir.out_.T

subplot(211)
plot(bh_out)
subplot(212)
plot(iir_out)
ylim(bh_out.min(), bh_out.max())
show()
