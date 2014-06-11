from brian2 import *
from brian2hears import *
import time

def test_indexing():
o    cf = erbspace(10*Hz, 1000*Hz, 32)
    signal = np.random.randn(1000)
    sound = TimedArray(signal, dt = defaultclock.dt)

    indices = np.zeros(2*len(cf), dtype = int)
    indices[::2] = np.arange(len(cf))
    indices[1::2] = np.arange(len(cf))

    iir_filter = GammatoneFilterbank(sound, cf)
    # multiplexed = NeuronGroup(2*len(cf), 'x_indices : integer')
    # multiplexed.variables.add_reference('x', iir_filter.variables['out'], index = 'x_indices')
    # multiplexed.x_indices = indices
    multiplexed = IndexedFilterbank(iir_filter, indices)
    Miir = StateMonitor(multiplexed, 'out', record = True)

    iir_net = Network(iir_filter, Miir)
    t0 = time.time()
    iir_net.run(len(signal)*defaultclock.dt)
    print 'IIR in ', time.time()-t0
    iir_out = Miir.out_.T

    assert (iir_out[:,1::2] == iir_out[:,::2]).all()


if __name__ == '__main__':
    test_indexing()
