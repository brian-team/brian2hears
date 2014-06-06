from brian2hears import *
from brian2 import *
import numpy as np

N = 1000
Ntaps = 20.
sound = TimedArray(np.random.randn(N), dt = defaultclock.dt)

iir_filter = IIRAveragingFilter(sound, Ntaps)
Miir = StateMonitor(iir_filter, 'out', record = True)
iir_net = Network(iir_filter, Miir)
iir_net.run(N*defaultclock.dt)
iir_out = Miir.out_.flatten()

fir_filter = FIRAveragingFilter(sound, Ntaps)
Mfir = StateMonitor(fir_filter, 'out', record = True)
fir_net = Network(fir_filter, Mfir)
fir_net.run(N*defaultclock.dt)
fir_out = Mfir.out_.flatten()

subplot(211)
plot(sound.values)
plot(fir_out)
subplot(212)
plot(iir_out)
plot(fir_out)
show()


