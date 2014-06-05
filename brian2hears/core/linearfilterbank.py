from brian2.core.variables import Variables
from brian2.core.base import BrianObject
from brian2.groups import Group, NeuronGroup
from brian2.synapses import Synapses
from brian2.utils import TimedArray
from brian2.units import Unit
from brian2.core.clocks import defaultclock

import numpy as np

class ShiftRegisterGroup(Group):
    '''
    A shift register

    Parameters
    ----------
    sound : (TimedArray)
        The time varying signal to be delayed

    Notes
    -----
    The ShiftRegisterGoup has an "out" Variable.

    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, sound, Ntaps, codeobj_class = None, when = None, name = 'shiftregistergroup*'):
        BrianObject.__init__(self, when = when, name = name)

        sr_equations = '''
        x : 1
        shift_indices : integer (constant)
        final_shift_indices : integer (constant)
        '''
        sr_g = NeuronGroup(Ntaps, model = sr_equations, 
                           codeobj_class = codeobj_class, clock = self.clock, namespace = {})
        sr_S = Synapses(sr_g, sr_g, 
                        codeobj_class = codeobj_class, clock = self.clock, 
                        namespace = {'sound':sound,
                                     'Ntaps':Ntaps})
        sr_S.connect(Ntaps - 1, Ntaps - 1)
        sr_custom_code = sr_S.runner('x_post = sound(t)', 
                                     when = (self.clock, 'start', 1))

        sr_g.shift_indices = np.roll(np.arange(Ntaps, dtype = int), -1)
        sr_g.variables.add_reference('shifted', sr_g.variables['x'], index = 'shift_indices')
        sr_g_custom_code = sr_g.runner('x = shifted', 
                                       when = (self.clock, 'start', 0))

        # add contained objects
        self._contained_objects += [sr_g, sr_S, sr_g_custom_code, sr_custom_code]

        # set up the variables
        self.variables = Variables(self)
        # this line gives the name of the output variable

        self.variables.add_array('final_shift_indices', Unit(1), Ntaps, dtype = int, constant = True)
        self.variables['final_shift_indices'].set_value(np.arange(Ntaps, dtype = int)[::-1])
        
        self.variables.add_reference('out', sr_g.variables['x'], index = 'final_shift_indices') # here should go the fancy indexing for Repeat/Tile etc.
        self.variables.add_constant('N', Unit(1), Ntaps) # a group has to have an N
        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N
