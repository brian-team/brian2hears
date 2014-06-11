from brian2hears import *
from brian2hears.library.gammatone import *
from brian2 import *
import numpy as np
import time
brian_prefs.codegen.target = 'weave'

if True:
    cf = erbspace(10*Hz, 1000*Hz, 32)
    signal = np.random.randn(1000)
    sound = TimedArray(signal, dt = defaultclock.dt)

    gt_filter = GammatoneFilterbank(sound, cf)
    a = gt_filter.filt_a
    b = gt_filter.filt_b
    f2 = FilterbankCascade(sound, b, a)
    
    M = StateMonitor(gt_filter, 'out', record = True)
    M2 = StateMonitor(f2, 'out', record = True)

    net = Network(gt_filter, f2, M, M2)
    net.run(len(signal)*defaultclock.dt)
    
    plot(M.out_.T)
    plot(M2.out_.T)
    show()
    

if False:
    # some test to demonstrate how its done
    f = open('./temp.npz', 'r')
    tmp = np.load(f)
    signal = tmp['signal']
    cf = tmp['cf']
    bh_out = tmp['bh_out']
    f.close()
    samplerate = 10000.
    defaultclock.dt = 1./samplerate*second
    sound = TimedArray(signal, dt = defaultclock.dt)

    iir_filter = GammatoneFilterbank(sound, cf)
    a, b = iir_filter.filt_a, iir_filter.filt_b

    f0 = LinearFilterbankGroup(sound, b[:,:,0], a[:,:,0])
    f1 = LinearFilterbankGroup(f0, b[:,:,1], a[:,:,1])
    f2 = LinearFilterbankGroup(f1, b[:,:,2], a[:,:,2])
    f3 = LinearFilterbankGroup(f2, b[:,:,3], a[:,:,3])

    M0 = StateMonitor(f0, 'out', record = True)
    M1 = StateMonitor(f1, 'out', record = True)
    M2 = StateMonitor(f2, 'out', record = True)
    M3 = StateMonitor(f3, 'out', record = True)

    net = Network(f0, f1, f2, f3, M0, M1, M2, M3)
    net.run(len(signal)*defaultclock.dt)

    subplot(411)
    plot(M3.out_.T)
    subplot(412)
    plot(M2.out_.T)
    subplot(413)
    plot(M1.out_.T)
    subplot(414)
    plot(M0.out_.T)

    figure()
    b2h_out = M3.out_.T
    plot(bh_out)
    plot(b2h_out)

    figure()
    plot(b2h_out-bh_out)

    show()





