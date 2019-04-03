import weakref

from brian2 import NeuronGroup, Clock, NetworkOperation, get_device, second
from brian2.devices.device import RuntimeDevice
from brian2.core.functions import timestep
from brian2.units.fundamentalunits import have_same_dimensions, DimensionMismatchError, DIMENSIONLESS


__all__ = ['FilterbankGroup']

class ApplyFilterbank(object):
    def __init__(self, group, targetvar, filterbank, buffersize):
        self.group = weakref.ref(group)
        self.targetvar = targetvar
        self.filterbank = weakref.proxy(filterbank)
        self.buffersize = buffersize
        self.dt = 1/filterbank.samplerate
        self.buffer_start = -2*buffersize
        self.buffer_end = -buffersize
        self.buffer = None

    def __call__(self, t):
        if not hasattr(self, 'target_variable'):
            self.target_variable = weakref.ref(self.group().variables[self.targetvar])
        i = timestep(t, self.dt)
        if not (self.buffer_start<=i<self.buffer_end):
            if i==0:
                self.filterbank.buffer_init()
            self.buffer_start = i
            self.buffer_end = self.buffer_start+self.buffersize
            self.buffer = self.filterbank.buffer_fetch(self.buffer_start, self.buffer_end)
        self.target_variable().set_value(self.buffer[i-self.buffer_start, :])


class FilterbankGroup(NeuronGroup):
    '''
    Allows a Filterbank object to be used as a NeuronGroup
    
    Initialised as a standard `~brian2.groups.neurongroup.NeuronGroup` object,
    but with two additional arguments at the beginning, and no ``N`` (number of
    neurons) argument.  The number of neurons in the group will be the number of
    channels in the filterbank.
    
    ``filterbank``
        The Filterbank object to be used by the group. In fact, any `.Bufferable`
        object can be used.
    ``targetvar``
        The target variable to put the filterbank output into.
        
    One additional keyword is available beyond that of
    `~brian2.groups.neurongroup.NeuronGroup`:
    
    ``buffersize=32``
        The size of the buffered segments to fetch each time. The efficiency
        depends on this in an unpredictable way, larger values mean more time
        spent in optimised code, but are worse for the cache. In many cases,
        the default value is a good tradeoff. Values can be given as a number
        of samples, or a length of time in seconds.
        
    Note that if you specify your own `~brian2.core.clocks.Clock`, it should
    have 1/dt=samplerate.
    '''
    
    def __init__(self, filterbank, targetvar, *args, **kwds):
        # Make sure we're not in standalone mode (which won't work)
        if not isinstance(get_device(), RuntimeDevice):
            raise RuntimeError("Cannot use standalone mode with brian2hears")

        self.targetvar = targetvar
        self.filterbank = filterbank
        filterbank.buffer_init()

        # Sanitize the clock - does it have the right dt value?
        if 'clock' in kwds:
            if int(1/kwds['clock'].dt)!=int(filterbank.samplerate):
                raise ValueError('Clock should have 1/dt=samplerate')
        elif 'dt' in kwds:
            if int(1 / kwds['dt']) != int(filterbank.samplerate):
                raise ValueError('Require 1/dt=samplerate')
        else:
            kwds['dt'] = 1/filterbank.samplerate
        
        buffersize = kwds.pop('buffersize', 32)
        if not isinstance(buffersize, int):
            if not have_same_dimensions(buffersize, second):
                raise DimensionMismatchError("buffersize argument should be an integer or in seconds")
            buffersize = int(buffersize*filterbank.samplerate)

        self.buffersize = buffersize

        self.apply_filterbank = ApplyFilterbank(self, targetvar, filterbank, buffersize)

        NeuronGroup.__init__(self, filterbank.nchannels, *args, **kwds)

        if self.variables[targetvar].dim is not DIMENSIONLESS:
            raise DimensionMismatchError("Target variable must be dimensionless")

        apply_filterbank_output = NetworkOperation(self.apply_filterbank.__call__, when='start', clock=self.clock)
        self.contained_objects.append(apply_filterbank_output)
