from brian2hears import *
from matplotlib import *
from brian2 import *

def test_delayfb():
    delays = np.arange(100)
    
    signal_array = np.zeros(1000)#np.random.randn(1000)
    signal_array[1] = 1

    fb = DelayFilterbank(TimedArray(signal_array, dt = defaultclock.dt), delays)

    M = StateMonitor(fb, 'out', record = True)

    net = Network(fb, M)
    net.run(len(signal_array)*defaultclock.dt)

    out = M.out_

    out_delay = np.zeros(out.shape[0])
    for k in range(out.shape[0]):
        out_delay[k] = np.min(np.nonzero(out[k,:])[0])

    assert (delays+1 == out_delay).all()

if __name__ == '__main__':
    test_delayfb()
    print 'Done'
