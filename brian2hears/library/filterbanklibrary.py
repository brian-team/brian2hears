from brian2hears.core.linearfilterbank import *
import numpy as np

class DelayFilterbank(FIRFilterbankGroup):
    '''
    A Filterbank that differentially delays input signals

    Parameters
    ----------
    sound
        As in the main FIRFilterbankGroup
    delays
        integer delays
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
        As in the main FIRFilterbankGroup
    Ntaps
        integer delays
    '''
    def __init__(self, sound, Ntaps):
        coeficients = np.ones((1, Ntaps))/float(Ntaps)

        FIRFilterbankGroup.__init__(self, sound, coeficients)

class IIRAveragingFilter(IIRFilterbankGroup):
    '''
    Parameters
    ----------
    sound
        As in the main FIRFilterbankGroup
    Ntaps
        integer delays
    '''
    def __init__(self, sound, Ntaps):
        b = np.array([0, 1/float(Ntaps)])
        a = np.array([1, -(1-1/float(Ntaps))])
        
        IIRFilterbankGroup.__init__(self, sound, b, a)
