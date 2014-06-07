from brian import *
from brian.hears import *
import time

N = 10000
signal = np.random.randn(N)
samplerate = 1./defaultclock.dt
cf = erbspace(100*Hz, 20000*Hz, 64)

sound = Sound(signal, samplerate = samplerate)
t0 = time.time()
bh_out = Gammatone(sound, cf).process()
print 'BH in ', time.time()-t0

f = open('./temp.npz', 'wb')
np.savez(f, signal=signal, cf = cf, bh_out = bh_out)
f.close()

