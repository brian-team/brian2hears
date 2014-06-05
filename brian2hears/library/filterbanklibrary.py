from brian2hears.core.firfilterbank import FIRFilterbankGroup
import numpy as np

__all__ = ['DelayFilterbank', 'GammatoneFilterbank']

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

class FIRGammatoneFilterbank(FIRFilterbankGroup):
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
        
