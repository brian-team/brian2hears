from brian2hears.core.linearfilterbank import *
from brian2 import *

def test_with_noise():
    s = np.random.randn(10000)
    g = ShiftRegisterGroup(TimedArray(s, dt = defaultclock.dt), 100)

    M = StateMonitor(g, 'out', record = True)

    run(1000*defaultclock.dt)
    
    out = M.out_
    
    out_delay = np.zeros(out.shape[0])
    for k in range(out.shape[0]):
        out_delay[k] = np.min(np.nonzero(out[k,:])[0])
        
    print out_delay, np.arange(100)[::-1]
    assert (np.arange(100)[::-1] == out_delay).all()

def test_with_click():
    s = np.zeros(10000)
    s[0:1] = 1.
    g = ShiftRegisterGroup(TimedArray(s, dt = defaultclock.dt), 100)

    M = StateMonitor(g, 'out', record = True)

    run(1000*defaultclock.dt)
    
    out = M.out_
    imshow(M.out_)
    return out        
    out_delay = np.zeros(out.shape[0])
    for k in range(out.shape[0]):
        out_delay[k] = np.min(np.nonzero(out[k,:])[0])

    print out_delay, np.arange(100)[::-1]
    assert (np.arange(100)[::-1] == out_delay).all()

def test_with_tone():
    times = np.arange(10000)*defaultclock.dt
    s = np.sin(2*np.pi*20.*Hz*times)

    g = ShiftRegisterGroup(TimedArray(s, dt = defaultclock.dt), 100)

    M = StateMonitor(g, 'out', record = True)

    run(1000*defaultclock.dt)
    
    out = M.out_
    imshow(M.out_)
    return out        
    out_delay = np.zeros(out.shape[0])
    for k in range(out.shape[0]):
        out_delay[k] = np.min(np.nonzero(out[k,:])[0])

    print out_delay, np.arange(100)[::-1]
    assert (np.arange(100)[::-1] == out_delay).all()

if __name__ == '__main__':
    out = test_with_click()
