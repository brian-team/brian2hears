from brian2hears.core.linearfilterbank import *
from brian2 import *

def test_with_noise():
    s = np.random.randn(10000)
    g = ShiftRegisterGroup(TimedArray(s, dt = defaultclock.dt), 100)

    M = StateMonitor(g, 'out', record = True)

    net = Network(M, g)
    net.run(1000*defaultclock.dt)

    out = M.out_
    
    out_delay = np.zeros(out.shape[0])
    for k in range(out.shape[0]):
        out_delay[k] = np.min(np.nonzero(out[k,:])[0])

    assert (np.arange(100)[::-1] == out_delay).all()

def test_with_click():
    s = np.zeros(10000)
    s[0] = 1.
    g = ShiftRegisterGroup(TimedArray(s, dt = defaultclock.dt), 100)

    M = StateMonitor(g, 'out', record = True)

    net = Network(M, g)
    net.run(1000*defaultclock.dt)
    
    out = M.out_
    imshow(M.out_)
    out_delay = np.zeros(out.shape[0])
    for k in range(out.shape[0]):
        out_delay[k] = np.min(np.nonzero(out[k,:])[0])
    assert (np.arange(100)[::-1] == out_delay).all()

def test_with_tone():
    times = np.arange(10000)*defaultclock.dt
    s = np.sin(2*np.pi*20.*Hz*times + 0.1)

    g = ShiftRegisterGroup(TimedArray(s, dt = defaultclock.dt), 100)

    M = StateMonitor(g, 'out', record = True)
    net = Network(M, g)
    net.run(1000*defaultclock.dt)
    
    out = M.out_

    out_delay = np.zeros(out.shape[0])
    for k in range(out.shape[0]):
        out_delay[k] = np.min(np.nonzero(out[k,:])[0])

    assert (np.arange(100)[::-1] == out_delay).all()

if __name__ == '__main__':
    test_with_click()
    test_with_tone()
    test_with_noise()
    print 'Done'

