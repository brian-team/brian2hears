import time
from brian2hears import *
from brian2 import *
import numpy as np
from scipy.signal import lfilter

brian_prefs.codegen.target = 'weave'

N = 1000
Ntaps = 20.
sound = TimedArray(np.random.randn(N), dt = defaultclock.dt)

alpha = 0.1
b = np.ones(2)#array([0, 1/float(Ntaps)])
a = np.ones(2)#array([1, -(1-1/float(Ntaps))])
a[1] = - alpha
b[0] = (1-alpha)
b[1] = 0.
sp_filtered = lfilter(b.flatten(), a.flatten(), sound.values)
if False:
    def my_lfilter(b, a, x):
        # implement the diff equations in 
        Ntaps = len(b)
        z = np.zeros(Ntaps-1)
        y = np.zeros(len(x))
        for m in range(1, len(x)):
            print m, Ntaps, len(x)
            y[m] = b[0]*x[m] + z[0]
            for n in range(0, Ntaps-2):
                z[n] = b[n+1]*x[m] + z[n+1] - a[n+1]*y[m]
            z[Ntaps-2] = b[Ntaps-1]*x[m] - a[Ntaps-1]*y[m]        
        return y
    my_sp_filtered = my_lfilter(b.flatten(), a.flatten(), sound.values)

    figure()
    plot(sp_filtered)
    plot(my_sp_filtered)
    show()

if True:
    a, b = a.reshape((1, 2)), b.reshape((1, 2))
    print a, b
    iir_filter = ExplicitIIRFilterbankGroup(sound, b, a)
    Miir = StateMonitor(iir_filter, 'out', record = [0])
    iir_net = Network(iir_filter, Miir)
    t0 = time.time()
    iir_net.run(N*defaultclock.dt)
    print 'IIR in ', time.time()-t0
    iir_out = Miir.out_.flatten()

    sp_filtered = lfilter(b.flatten(), a.flatten(), sound.values)

    plot(sound.values)

    plot(iir_out)
    plot(sp_filtered)
    show()

if True:
    iir_filter = ExplicitIIRAveragingFilter(sound, Ntaps)
    Miir = StateMonitor(iir_filter, 'out', record = [0])
    iir_net = Network(iir_filter, Miir)
    t0 = time.time()
    iir_net.run(N*defaultclock.dt)
    print 'IIR in ', time.time()-t0
    iir_out = Miir.out_.flatten()

    fir_filter = FIRAveragingFilter(sound, Ntaps)
    Mfir = StateMonitor(fir_filter, 'out', record = True)
    fir_net = Network(fir_filter, Mfir)
    t0 = time.time()
    fir_net.run(N*defaultclock.dt)
    print 'FIR in ', time.time()-t0
    fir_out = Mfir.out_.flatten()

    sp_filtered = lfilter(iir_filter.b.flatten(), iir_filter.a.flatten(), sound.values)

    subplot(211)
    plot(sound.values)
    plot(sp_filtered)
    plot(fir_out)

    subplot(212)
    plot(iir_out)
    plot(sp_filtered)
    show()


