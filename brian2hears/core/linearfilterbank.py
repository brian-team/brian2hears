from brian2.core.variables import Variables
from brian2.core.base import BrianObject
from brian2.groups import Group, NeuronGroup
from brian2.synapses import Synapses
from brian2.utils import TimedArray
from brian2.units import Unit
from brian2.core.clocks import defaultclock

#from brian2hears.core.filterbank import IndexedFilterbank, FunctionFilterbank

import numpy as np

class ShiftRegister(Group):
    '''
    A shift register

    Parameters
    ----------
    sound : (TimedArray)
        The time varying signal to be delayed

    Keywords
    --------
    reverse_output : (bool)
        If set to True, then the output is in reverse order (but it requires one less reference indexing).
        It is True. This is convienient because of the shape of fir filter coeficients, so this may change.
        
    Notes
    -----
    The ShiftRegisterGoup has an "out" Variable.
    '''

    def __init__(self, sound, Ntaps, codeobj_class = None, when = None, name = 'shiftregistergroup*', reverse_output = True):
        # Shift-register group
        # 
        # The sound is written (thanks to the synapses sr_S) to the last neuron of the neurongroup, 
        # and a custom code operation is run within sr_g that shifts all values 
        # (i.e. does sr_g[i] = sr_g[i-1] for all i)
        # This makes use of the new indexing scheme: so it relies on the add_reference thing.

        BrianObject.__init__(self, when = when, name = name)

        # something to shift all values
        sr_equations = '''
        x : 1
        shift_indices : integer (constant)
        final_shift_indices : integer (constant)
        '''
        sr_g = NeuronGroup(Ntaps, model = sr_equations, 
                           codeobj_class = codeobj_class, clock = self.clock, namespace = {})
        sr_g.shift_indices = np.roll(np.arange(Ntaps, dtype = int), -1)
        sr_g.variables.add_reference('shifted', sr_g, 'x', index = 'shift_indices')
        sr_g_custom_code = sr_g.runner('x = shifted', 
                                       when = (self.clock, 'start', 0))

        # something to input the sound
        sr_S = Synapses(sr_g, sr_g, 
                        codeobj_class = codeobj_class, clock = self.clock, 
                        namespace = {'sound':sound,
                                     'Ntaps':Ntaps})
        sr_S.connect(Ntaps - 1, Ntaps - 1)
        sr_custom_code = sr_S.runner('x_post = sound(t)', 
                                     when = (self.clock, 'start', 1))



        # add contained objects
        self._contained_objects += [sr_g, sr_S, sr_g_custom_code, sr_custom_code]

        # set up the variables
        self.variables = Variables(self)
        self.variables.add_constant('N', Unit(1), Ntaps) # a group has to have an N
        self.variables.add_constant('start', Unit(1), 0) 
#        self.variables.add_constant('N', Unit(1), Ntaps) 
        self.variables.add_arange('i', Ntaps)

        # the output variable
        if reverse_output:
            self.variables.add_reference('out', sr_g, 'x')
        else:
            self.variables.add_array('final_shift_indices', Unit(1), Ntaps, dtype = int, constant = True)
            self.variables['final_shift_indices'].set_value(np.arange(Ntaps, dtype = int)[::-1])
            self.variables.add_reference('out', sr_g, 'x', index = 'final_shift_indices') # here should go the fancy indexing for Repeat/Tile etc.
        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N

class FIRFilterbank(Group):
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
    The `FIRFilterbank` has an "out" Variable that can be therefore
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

        sr_g = ShiftRegister(sound, Ntaps, 
                                  codeobj_class = None, 
                                  when = None, 
                                  name = 'shiftregistergroup*', 
                                  reverse_output = True)
        # Convolution with Synapses
        # 
        # out_g is a NeuronGroup that only holds the output variable (locally called "filtered").
        # There are Synapses from sr_g to out_g, each output channel has Ntaps synapses to sr_g: 
        # as a consequence, doing the weighed sum is just using a "(summed)" flag, and put the FIR coeficients 
        # in the right order.
        out_g = NeuronGroup(Nchannels, 'filtered : 1', 
                            codeobj_class = codeobj_class, clock = self.clock, namespace = {})
        fir_S = Synapses(sr_g, out_g, '''coef : 1
                                        filtered_post = out_pre*coef : 1 (summed)''', 
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
        self._contained_objects += [sr_g, out_g, fir_S]

        # set up the variables
        self.variables = Variables(self)
        # this line gives the name of the output variable
        self.variables.add_reference('out', out_g, 'filtered') # here goes the fancy indexing for Repeat/Tile etc.
        self.variables.add_constant('N', Unit(1), Nchannels) # a group has to have an N
        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N

class LinearFilterbank(Group):
    '''
    A filterbank of Infinite Impulse Response (IIR) filters.

    Parameters
    ----------
    source : (TimedArray or LinearFilterbankGroup)
        The time varying signal to which the filters are applied.
    a, b : (TimedArray)
        Matrices of size (Nchannels, Ntaps) giving the values of the a and b IIR coeficients
        

    Generalised linear filterbank
    
    Initialisation arguments:

    ``source``
        The input to the filterbank, must have the same number of channels or
        just a single channel. In the latter case, the channels will be
        replicated.
    ``b``, ``a``
        The coeffs b, a must be of shape ``(nchannels, m)`` or
        ``(nchannels, m, p)``. Here ``m`` is
        the order of the filters, and ``p`` is the number of filters in a
        chain (first you apply ``[:, :, 0]``, then ``[:, :, 1]``, etc.).
    
    The filter parameters are stored in the modifiable attributes ``filt_b``,
    ``filt_a`` and ``filt_state`` (the variable ``z`` in the section below).
    
    **Notes**
    
    These notes adapted from scipy's :func:`~scipy.signal.lfilter` function.
    
    The filterbank is implemented as a direct II transposed structure.
    This means that for a single channel and element of the filter cascade,
    the output y for an input x is defined by::

        a[0]*y[m] = b[0]*x[m] + b[1]*x[m-1] + ... + b[m]*x[0]
                              - a[1]*y[m-1] - ... - a[m]*y[0]

    using the following difference equations::

    y[m] = b[0]*x[m] + z[0,m-1]
    z[0,m] = b[1]*x[m] + z[1,m-1] - a[1]*y[m]
    ...
    z[n-3,m] = b[n-2]*x[m] + z[n-2,m-1] - a[n-2]*y[m]
    z[n-2,m] = b[n-1]*x[m] - a[n-1]*y[m]

    where m is the output sample number.

    The rational transfer function describing this filter in the
    z-transform domain is::
    
                                -1              -nb
                    b[0] + b[1]z  + ... + b[m] z
            Y(z) = --------------------------------- X(z)
                                -1              -na
                    a[0] + a[1]z  + ... + a[m] z

    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, source, b, a, 
                 codeobj_class = None, when = None, name = 'linearfilterbankgroup*'):

        BrianObject.__init__(self, when = when, name = name)

        Nchannels, Ntaps = b.shape[0], b.shape[1]

        if isinstance(source, TimedArray):
            sound = source
            source_is_fb = False
        elif isinstance(source, Group):
            source = source
            source_is_fb = True

        self.filt_a = a
        self.filt_b = b
#        assert (self.filt_a.ndim == 2) and (self.filt_b.ndim == 2)

        if source_is_fb:
            z_equations = ''#x : 1 \n'
        else:
            z_equations = 'x : 1 (scalar) \n'

        z_equations += 'y : 1 \n'
        for ktap in range(Ntaps):
            z_equations += 'z%d_m : 1 \n' % ktap
            z_equations += 'a_%d : 1 (constant) \n' % ktap
            z_equations += 'b_%d : 1 (constant) \n' % ktap

        
        z_updates = ''
        if not source_is_fb:
            z_updates += 'x = sound(t) \n'
            main_namespace =  {'sound': sound}
        else:
            main_namespace =  {}


        z_updates += 'y=b_0*x+z0_m \n'
        for kupdate in range(0, Ntaps-2):
            z_updates += 'z%d_m = b_%d*x + z%d_m - a_%d*y \n' % (kupdate, kupdate+1, kupdate+1, kupdate+1) 
        z_updates += 'z%d_m = b_%d*x- a_%d*y \n' % (Ntaps-2, Ntaps-1, Ntaps - 1)

        main_group = NeuronGroup(Nchannels, model = z_equations, 
                          codeobj_class = codeobj_class, clock = self.clock, namespace = main_namespace)
        main_group_updates = main_group.runner(z_updates,
                                                   when = (self.clock, 'start', 0))
        
        for k in range(Ntaps):
            exec('main_group.a_%d = a[:,%d]' % (k,k))
            exec('main_group.b_%d = b[:,%d]' % (k,k))

        if source_is_fb:
            main_group.variables.add_reference('x', source, 'out')
            

        ####################################
        # Brian Group infrastructure stuff #
        ####################################
        # add contained objects
        self._contained_objects += [main_group, main_group_updates]#main_group_compute_y, 

        # set up the variables
        self.variables = Variables(self)
        # this line gives the name of the output variable

        self.variables.add_reference('out', main_group, 'y') 

        self.variables.add_constant('N', Unit(1), Nchannels) # a group has to have an N
        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()


    def __len__(self):
        # this should probably not be needed in the future
        return self.N


class OldIIRFilterbankGroup(Group):
    '''
    A filterbank of Infinite Impulse Response (IIR) filters.

    Parameters
    ----------
    sound : (TimedArray)
        The time varying signal to which the filters are applied.
    fir_coeficients : (TimedArray)
        A matrix of size (Nchannels, Ntaps) giving the values of the
    impulse response in each channel as a function of time.


    Generalised linear filterbank
    
    Initialisation arguments:

    ``source``
        The input to the filterbank, must have the same number of channels or
        just a single channel. In the latter case, the channels will be
        replicated.
    ``b``, ``a``
        The coeffs b, a must be of shape ``(nchannels, m)`` or
        ``(nchannels, m, p)``. Here ``m`` is
        the order of the filters, and ``p`` is the number of filters in a
        chain (first you apply ``[:, :, 0]``, then ``[:, :, 1]``, etc.).
    
    The filter parameters are stored in the modifiable attributes ``filt_b``,
    ``filt_a`` and ``filt_state`` (the variable ``z`` in the section below).
    
    Has one method:
    
    .. automethod:: decascade
    

    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, sound, b, a, 
                 codeobj_class = None, when = None, name = 'iirfilterbankgroup*'):
        BrianObject.__init__(self, when = when, name = name)

        Ntaps = b.shape[0]

        z_equations = '''
        x : 1 (scalar)
        z : 1
        y : 1
        '''
        main_group = NeuronGroup(Ntaps, model = z_equations, 
                          codeobj_class = codeobj_class, clock = self.clock, namespace = {'sound': sound})
        # First, write the signal value to the x scalar
        main_group_write_sound = main_group.runner('x = sound(t)',
                                     when = (self.clock, 'start', 0))

        # second, compute y[m]
#        main_group_compute_y = main_group.runner('y = b0 * x + z0m1', 
#                                                 when = (self.clock, 'start', 1))
#        main_group.b0 = b[0]



        # third, compute all Ntaps-1 difference equations: 
        # z[i,m] = b[i+1]*x[m] + z[i+1,m-1] - a[i+1]*y[m]
        # i goes in order from 0 to Ntaps - 2 (last equation is different, see above
        S_model = '''
zflag : 1
a : 1
b : 1
'''
        S_diffeq = Synapses(main_group, main_group, model = S_model,
                            codeobj_class = codeobj_class, clock = self.clock, 
                            namespace = {})
        #'(i + 1 == j) and (j != Ntaps - 1)') # explicitly should be better I suppose
        S_diffeq.connect(np.roll(np.arange(Ntaps), -1), np.arange(Ntaps))
        S_diffeq_code = S_diffeq.runner('z_post = b * x_pre + z_pre*zflag - a * y_pre',
                                        when = (self.clock, 'start', 2))
        S_diffeq.a = a
#        S_diffeq.a[0] = 0
        S_diffeq.b = b

        # To implement the last equation
        # z[Ntaps-2,m] = b[Ntaps-1]*x[m] - a[Ntaps-1]*y[m]
        # I first have to make sure that the term in the diffeq with z[Ntaps-1,m-1] (i.e. z_pre[Ntaps-1]) vanishes
        # this is what zflag does
        flags = np.ones(Ntaps)
        flags[-1] = 0
        S_diffeq.zflag = flags


#        S_write_z0m1 = Synapses(main_group, main_group,
#                                codeobj_class = codeobj_class, clock = self.clock, 
#                                namespace = {})
#        S_write_z0m1.connect(np.arange(Ntaps-1), 0)
#        S_write_z0m1_code = S_write_z0m1.runner('z0m1 = z_post',
#                                                when = (self.clock, 'start', 3))
        
        ####################################
        # Brian Group infrastructure stuff #
        ####################################
        # add contained objects
        self._contained_objects += [main_group, main_group_write_sound]#main_group_compute_y, 
        self._contained_objects += [S_diffeq_code, S_diffeq]
#        self._contained_objects += [S_write_z0m1_code, S_write_z0m1]

        # set up the variables
        self.variables = Variables(self)
        # this line gives the name of the output variable
        self.variables.add_reference('out', main_group, 'z') # here goes the fancy indexing for Repeat/Tile etc.
#        self.variables.add_reference('out', main_group.variables['x']) # here goes the fancy indexing for Repeat/Tile etc.
        self.variables.add_constant('N', Unit(1), Ntaps) # a group has to have an N
        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N
