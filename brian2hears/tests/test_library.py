from brian2 import *
from brian2hears import *

if False:
    # test the approximate gammatone
    cfs = erbspace(10*Hz, 1000*Hz, 32)
    sound = TimedArray(np.random.randn(1000), dt = defaultclock.dt)
    g = ApproximateGammatone(sound, cfs)
    M = StateMonitor(g, 'out', record = True)

    net = Network(g, M)
    net.run(len(sound.values)*defaultclock.dt)

    plot(M.out_.T)

if False:
    # test the log gammachirp
    cfs = erbspace(10*Hz, 1000*Hz, 32)
    sound = TimedArray(np.random.randn(1000), dt = defaultclock.dt)
    g = LogGammachirp(sound, cfs)
    M = StateMonitor(g, 'out', record = True)

    net = Network(g, M)
    net.run(len(sound.values)*defaultclock.dt)

    plot(M.out_.T)

if True:
    # this is FIR and therefore should not work
    # test the linear gammachirp
    cfs = erbspace(10*Hz, 1000*Hz, 32)
    sound = TimedArray(np.random.randn(1000), dt = defaultclock.dt)
    g = LinearGammachirp(sound, cfs)
    M = StateMonitor(g, 'out', record = True)

    net = Network(g, M)
    net.run(len(sound.values)*defaultclock.dt)

    plot(M.out_.T)
