from brian2.core.variables import Variables
from brian2.core.base import BrianObject
from brian2.groups import Group, NeuronGroup
from brian2.synapses import Synapses
from brian2.utils import TimedArray
from brian2.units import Unit
from brian2.core.clocks import defaultclock

import numpy as np

__all__ = ['FIRFilterbankGroup']

class FIRFilterbankGroup(Group):
    '''
    A filterbank of Finite Impulse Response (FIR) filters.

    Parameters
    ----------
    sound : (TimedArray)
        The time varying signal to which the filters are applied.
    fir_coeficients : (TimedArray)
        A matrix of size (Nchannels, Ntaps) giving the values of the
    impulse response in each channel as a function of time.

    Notes
    -----
    The FIRFilterbankGroup has an "out" Variable that can be therefore
    used to be linked to another Brian NeuronGroup object for example.
    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, sound, fir_coeficients, 
                 codeobj_class = None, when = None, name = 'firfilterbankgroup*'):
        BrianObject.__init__(self, when = when, name = name)

        # FIR filtering specific stuff
        Nchannels, Ntaps = fir_coeficients.shape[0], fir_coeficients.shape[1]

        ####################################
        #      Convolution in Brian2       #
        ####################################
        # 
        # The idea is just a straightforward implementation of convolution.
        # One neurongroup (the sr_g group) is in fact a shift_register, and a Synapses object 
        # then gets the time-shifted values, multiply them by the FIR filter coeficients, and 
        # return the sum of the values.
        #
        # Shift-register group
        # 
        # The sound is written (thanks to the synapses sr_S) to the last neuron of the neurongroup, 
        # and a custom code operation is run within sr_g that shifts all values 
        # (i.e. does sr_g[i] = sr_g[i-1] for all i)
        # This makes use of the new indexing scheme: so it relies on the add_reference thing.
        sr_equations = '''
        x : 1
        x_coef : 1
        shift_indices : integer (constant)
        coeficient : 1 (constant)
        '''
        sr_g = NeuronGroup(Ntaps, model = sr_equations, 
                           codeobj_class = codeobj_class, clock = self.clock, namespace = {})
        sr_g.coeficient = fir_coeficients

        sr_S = Synapses(sr_g, sr_g, 
                        codeobj_class = codeobj_class, clock = self.clock, 
                        namespace = {'sound':sound,
                                     'Ntaps':Ntaps})
        sr_S.connect(Ntaps - 1, Ntaps - 1)#Ntaps-1,Ntaps-1)
        sr_custom_code = sr_S.runner('x_post = sound(t+Ntaps*dt)', 
                                     when = (self.clock, 'start', 1))


        sr_g.shift_indices = np.roll(np.arange(Ntaps, dtype = int), -1)
        sr_g.variables.add_reference('shifted', sr_g.variables['x'], index = 'shift_indices')
        sr_g_custom_code = sr_g.runner('x = shifted', 
                                       when = (self.clock, 'start', 1))

        # Convolution with Synapses
        # 
        # out_g is a NeuronGroup that only holds the output variable (locally called "filtered").
        # There are Synapses from sr_g to out_g, each output channel has Ntaps synapses to sr_g: 
        # as a consequence, doing the weighed sum is just using a "(summed)" flag, and put the FIR coeficients 
        # in the right order.
        out_g = NeuronGroup(Nchannels, 'filtered : 1', 
                            codeobj_class = codeobj_class, clock = self.clock, namespace = {})
        fir_S = Synapses(sr_g, out_g, '''coef : 1
                                        filtered_post = x_pre*coef : 1 (summed)''', 
                         codeobj_class = codeobj_class, clock = self.clock, namespace = {})
        # connect the Synapses
        for k in range(Nchannels):
            fir_S.connect(np.arange(Ntaps), k)
        # put in filter coeficients 
        fir_S.coef = fir_coeficients[:,::-1].flatten()

        ####################################
        # Brian Group infrastructure stuff #
        ####################################

        # add contained objects
        self._contained_objects += [sr_g, out_g, sr_S, fir_S, sr_g_custom_code, sr_custom_code]

        # set up the variables
        self.variables = Variables(self)
        # this line gives the name of the output variable
        self.variables.add_reference('out', out_g.variables['filtered']) # here goes the fancy indexing for Repeat/Tile etc.
        self.variables.add_constant('N', Unit(1), Nchannels) # a group has to have an N
        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N


if __name__ == '__main__':
    from brian2.monitors import StateMonitor
    from brian2.core.magic import run
    from brian2.units.stdunits import ms
    from matplotlib.pyplot import *

    Nchannels = 100
    Ntaps = 1024
    
    x = TimedArray(np.random.randn(10000), dt=defaultclock.dt)
    
    coeficients = np.zeros((Nchannels, Ntaps), dtype = float)
    for k in range(Nchannels):
        coeficients[k, 64+k] = 1.
        
    g = FIRFilterbankGroup(x, coeficients)
    print "o"
    M = StateMonitor(g, 'out', record = True)
    
    run(100*ms)
    
    imshow(M.out_)
    show()
        

    
