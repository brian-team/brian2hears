from brian2.core.variables import Variables
from brian2.core.base import BrianObject
from brian2.groups import Group, NeuronGroup
from brian2.synapses import Synapses
from brian2.utils import TimedArray
from brian2.units import Unit
from brian2.core.clocks import defaultclock

from brian2hears.core.linearfilterbank import LinearFilterbank

import numpy as np

class IndexedFilterbank(Group):
    '''
    Formerly "RestructureFilterbank"
    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, source, shift_indices, dt=None, when='start', order=0,
                 clock=None, name = 'restructurefilterbank*'):
        BrianObject.__init__(self, dt=dt, when=when, order=order, clock=clock,
                             name=name)

        Nout = len(shift_indices)

        # set up the variables
        self.variables = Variables(self)
        self.variables.add_constant('N', Unit(1), Nout) # a group has to have an N
        self.variables.add_constant('start', Unit(1), 0) 
        self.variables.add_arange('i', Nout)

        # here is where the magic happens
        self.variables.add_array('shift_indices', Unit(1), Nout, dtype = int, constant = True)
        self.variables['shift_indices'].set_value(shift_indices)

        self.variables.add_reference('out', source, 'out', index = 'shift_indices')

        self.variables.create_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N

class Repeat(IndexedFilterbank):
    '''
    * Repeating

    LLLRRR
    '''

    def __init__(self, source, N, dt=None, when='start', order=0, clock=None,
                 name='repeatfilterbank*'):
        indices = np.repeat(np.arange(len(source), dtype = int), N)
        IndexedFilterbank.__init__(self, source, indices, dt=dt, when=when,
                                   order=order, clock=clock, name=name)

class Tile(IndexedFilterbank):
    '''
    * Tiling 

    LRLRLR
    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, source, N, dt=None, when='start', order=0, clock=None,
                 name='tilefilterbank*'):
        indices = np.tile(np.arange(len(source), dtype = int), N)
        IndexedFilterbank.__init__(self, source, indices, dt=dt, when=when,
                                   order=order, clock=clock, name = name)

class FunctionFilterbank(Group):
    '''
    * FunctionFilterbank * 

    Applies a static function to the output of a filterbank.
    The syntax has changed, now the function must be input as a string, where 'x' corresponds to the output of the source filterbank.
    Much as other filterbanks for now, it has a variable 'out' which contains the output.
    
    An example usage::
    >>> func = FunctionFilterbank(source_filterbank, 'clip(x, 0, Inf)')
    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, source, function_statement, dt=None, when='start', order=0,
                 clock=None, name = 'functionfilterbank*'):
        BrianObject.__init__(self, dt=dt, when=when, order=order, clock=clock,
                             name=name)

        Nchannels = len(source)

        g = NeuronGroup(source.N, model='y:1', clock=self.clock, namespace={})
        g.variables.add_reference('x', source, 'out')
        custom_code = g.custom_operation('y = '+function_statement)

        self._contained_objects += [g, custom_code]

        # set up the variables
        self.variables = Variables(self)
        self.variables.add_constant('N', Unit(1), Nchannels) # a group has to have an N
        self.variables.add_constant('start', Unit(1), 0) 
        self.variables.add_arange('i', Nchannels)

        self.variables.add_reference('out', g, 'y')

        self.variables.create_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N            

class FilterbankCascade(Group):
    '''
    A cascade of filters
    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, source, b, a,
                 dt=None, when='start', order=0, clock=None,
                 name='filterbankcascade*'):
        BrianObject.__init__(self, dt=dt, when=when, order=order, clock=clock,
                             name=name)

        Nchannels = a.shape[0]
        Nchain = a.shape[2]

        # deal with chaining the filterbank
        self.filt_a = a
        self.filt_b = b
        
        filters = [LinearFilterbank(source, self.filt_b[:,:,0], self.filt_a[:,:,0])]
        for k in range(1, Nchain):
            filters += [LinearFilterbank(filters[-1], self.filt_b[:,:,k], self.filt_a[:,:,k])]
        
        ####################################
        # Brian Group infrastructure stuff #
        ####################################
        # add contained objects
        self._contained_objects += filters

        # set up the variables
        self.variables = Variables(self)

        # this line gives the name of the output variable
        self.variables.add_reference('out', filters[-1], 'out') # here goes the fancy indexing for Repeat/Tile etc.

        self.variables.add_constant('N', Unit(1), Nchannels) # a group has to have an N
        self.variables.create_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N

