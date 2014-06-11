from brian2 import *
from brian2hears import *
import time


def test_functionfb():
    cf = erbspace(10*Hz, 1000*Hz, 16)
    signal = np.random.randn(1000)
    sound = TimedArray(signal, dt = defaultclock.dt)

    iir_filter = GammatoneFilterbank(sound, cf)
    func = FunctionFilterbank(iir_filter, 'clip(x, 0, Inf)')
    M = StateMonitor(func, 'out', record = True)

    iir_net = Network(iir_filter, M, func)
    t0 = time.time()
    iir_net.run(len(signal)*defaultclock.dt)
    print 'IIR in ', time.time()-t0
    iir_out = M.out_.T

    assert (iir_out >=0).all()
    #plot(iir_out[:,::2])
    #show()

if __name__ == '__main__':
    test_functionfb()
