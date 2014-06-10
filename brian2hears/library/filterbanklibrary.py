from brian2hears.core.linearfilterbank import *
import numpy as np

class DelayFilterbank(FIRFilterbankGroup):
    '''
    An FIR Filterbank that differentially delays input signals. 

    Parameters
    ----------
    `sound`
        As in LinearFilterbankGroup
    delays
        integer delays in samples
    '''
    def __init__(self, sound, delays):
        Ntaps = delays.max()+1
        Nchannels = len(delays)

        coeficients = np.zeros((Nchannels, Ntaps))
        for k in range(Nchannels):
            coeficients[k, delays[k]] = 1.

        FIRFilterbankGroup.__init__(self, sound, coeficients)

class FIRAveragingFilter(FIRFilterbankGroup):
    '''
    Parameters
    ----------
    sound
        As in LinearFilterbankGroup
    Ntaps
        Number of taps to average over
    '''
    def __init__(self, sound, Ntaps):
        coeficients = np.ones((1, Ntaps))/float(Ntaps)

        FIRFilterbankGroup.__init__(self, sound, coeficients)

class IIRAveragingFilter(LinearFilterbankGroup):
    '''
    An implementation of an IIR filter that performs averaring. It should be the IIR formulation of the FIR filter above, but I'm not so sure it is

    Parameters
    ----------
    source
        As in LinearFilterbankGroup
    Ntaps
        Number of taps to average over (the equivalent of, this filterbank is IIR with 2 taps)
    '''
    def __init__(self, sound, Ntaps, method = 'explicit'):
        self.b = np.array([0., 1/float(Ntaps)]).reshape((1,2))
        self.a = np.array([1., -(1-1/float(Ntaps))]).reshape((1,2))
        
        LinearFilterbankGroup.__init__(self, sound, self.b, self.a)



