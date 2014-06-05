'''
Adapted from the original brian.hears (which was adapted from MAP)
'''
from brian2.units import check_units
from brian2.units.stdunits import Hz, kHz
import numpy as np

__all__ = ['erbspace']

@check_units(low=Hz, high=Hz)
def erbspace(low, high, N, earQ=9.26449, minBW=24.7, order=1):
    '''
    Returns the centre frequencies on an ERB scale.
    
    ``low``, ``high``
        Lower and upper frequencies
    ``N``
        Number of channels
    ``earQ=9.26449``, ``minBW=24.7``, ``order=1``
        Default Glasberg and Moore parameters.
    '''
    low = float(low)
    high = float(high)
    cf = -(earQ * minBW) + np.exp((np.arange(N)) * (-np.log(high + earQ * minBW) + \
                                                    np.log(low + earQ * minBW)) / (N-1)) * (high + earQ * minBW)
    cf = cf[::-1]
    return cf * Hz

# Testing
if __name__ == '__main__':
    from matplotlib.pyplot import *
    cf = erbspace(20 * Hz, 20 * kHz, 3000)
    print np.amin(cf), np.amax(cf)
    print np.diff(cf)[-5:]
    plot(cf)
    show()
