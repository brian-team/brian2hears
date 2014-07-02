from brian2 import *
from brian2hears import *
from matplotlib import cm

N = 1000
signal = np.random.randn(N)
defaultclock.dt = 0.1*ms
sound = TimedArray(signal, dt = defaultclock.dt)

Ncf = 50
cf = np.array(erbspace(20*Hz, 20000.*Hz, Ncf))

iir_filter = GammatoneFilterbank(sound, cf)

Miir = StateMonitor(iir_filter, 'out', record = True)

iir_net = Network(iir_filter, Miir)

iir_net.run(len(signal)*defaultclock.dt)

iir_out = Miir.out_

cgram = iir_out#/np.max(iir_out, axis = 1).reshape(iir_out.shape[0], 1)
pcolor(np.arange(N), np.arange(Ncf), cgram, cmap = cm.coolwarm)

show()



