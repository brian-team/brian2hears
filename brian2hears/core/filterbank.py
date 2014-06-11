from brian2.core.variables import Variables
from brian2.core.base import BrianObject
from brian2.groups import Group, NeuronGroup
from brian2.synapses import Synapses
from brian2.utils import TimedArray
from brian2.units import Unit
from brian2.core.clocks import defaultclock

class IndexedFilterbank(Group):
    '''
    Formerly "RestructureFilterbank"
    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, source, shift_indices, when = None, name = 'restructurefilterbank*'):
        BrianObject.__init__(self, when = when, name = name)

        Nout = len(shift_indices)

        # set up the variables
        self.variables = Variables(self)
        self.variables.add_constant('N', Unit(1), Nout) # a group has to have an N
        self.variables.add_constant('start', Unit(1), 0) 
        self.variables.add_arange('i', Nout)

        # here is where the magic happens
        self.variables.add_array('shift_indices', Unit(1), Nout, dtype = int, constant = True)
        self.variables['shift_indices'].set_value(shift_indices)

        self.variables.add_reference('out', source.variables['out'], index = 'shift_indices')

        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N

class Repeat(IndexedFilterbank):
    def __init__(self, source, N, when = None, name = 'repeatfilterbank*'):
        indices = np.zeros(len(source)*N, dtype = int)
        for k in range(N):
            indices[k::N] = np.arange(len(source), dtype = int)
        IndexedFilterbank.__init__(self, source, indices, when = when, name = name)

class Tile(IndexedFilterbank):
    

    def __init__(self, source, N, when = None, name = 'tilefilterbank*'):
        indices = np.tile(np.arange(len(source), dtype = int), N)
        IndexedFilterbank.__init__(self, source, indices, when = when, name = name)

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
    def __init__(self, source, function_statement, when = None, name = 'functionfilterbank*'):
        BrianObject.__init__(self, when = when, name = name)

        Nchannels = len(source)

        g = NeuronGroup(source.N, model = 'y:1', clock = self.clock, namespace = {})
        g.variables.add_reference('x', source.variables['out'])
        custom_code = g.runner('y = '+function_statement)

        self._contained_objects += [g, custom_code]

        # set up the variables
        self.variables = Variables(self)
        self.variables.add_constant('N', Unit(1), Nchannels) # a group has to have an N
        self.variables.add_constant('start', Unit(1), 0) 
        self.variables.add_arange('i', Nchannels)

        self.variables.add_reference('out', g.variables['y'])

        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N            
