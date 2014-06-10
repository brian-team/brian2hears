from brian2hears.core.linearfilterbank import LinearFilterbankGroup
from brian2.core.variables import Variables
from brian2.core.base import BrianObject
from brian2.groups import Group
from brian2.units import Unit
from brian2.core.clocks import defaultclock

import numpy as np

class GammatoneFilterbank(Group):
    '''
    A Gammatone filterbank consisting of len(cf) channels.
    The implementation uses the LinearFilterbankGroup and the filter coeficients as in brian.hears.
    
    The filterbank has a variable 'out' which holds the output of the filterbank.

    Bank of gammatone filters.
    
    They are implemented as cascades of four 2nd-order IIR filters (this
    8th-order digital filter corresponds to a 4th-order gammatone filter).
    
    The approximated impulse response :math:`\\mathrm{IR}` is defined as follow
    :math:`\\mathrm{IR}(t)=t^3\\exp(-2\\pi b \\mathrm{ERB}(f)t)\\cos(2\\pi f t)`
    where :math:`\\mathrm{ERB}(f)=24.7+0.108 f` [Hz] is the equivalent
    rectangular bandwidth of the filter centered at :math:`f`.

    It comes from Slaney's exact gammatone implementation (Slaney, M., 1993,
    "An Efficient Implementation of the Patterson-Holdsworth 
    Auditory Filter Bank". Apple Computer Technical Report #35). The code is
    based on
    `Slaney's Matlab implementation <http://cobweb.ecn.purdue.edu/~malcolm/interval/1998-010/>`__.
    
    Initialised with arguments:
    
    ``source``
        a sound in the TimedArray format
        
    ``cf``
        List or array of center frequencies.
        
    ``b=1.019``
        parameter which determines the bandwidth of the filters (and
        reciprocally the duration of its impulse response). In particular, the
        bandwidth = b.ERB(cf), where ERB(cf) is the equivalent bandwidth at
        frequency ``cf``. The default value of ``b`` to a best fit
        (Patterson et al., 1992). ``b`` can either be a scalar and will be the
        same for every channel or an array of the same length as ``cf``.
        
    ``erb_order=1``, ``ear_Q=9.26449``, ``min_bw=24.7``
        Parameters used to compute the ERB bandwidth.
        :math:`\\mathrm{ERB} = ((\mathrm{cf}/\mathrm{ear\\_Q})^{\\mathrm{erb}\\_\\mathrm{order}} + \\mathrm{min\\_bw}^{\\mathrm{erb}\\_\\mathrm{order}})^{(1/\\mathrm{erb}\\_\\mathrm{order})}`.
        Their default values are the ones recommended in
        Glasberg and Moore, 1990. 

    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, sound, cf, b=1.019, erb_order=1, ear_Q=9.26449,
                 min_bw=24.7,
                 codeobj_class = None, when = None, name = 'gammatonefilterbank*'):
        BrianObject.__init__(self, when = when, name = name)

        cf = np.atleast_1d(cf)
        Nchannels = len(cf)

        T = float(defaultclock.dt)# samplerate goes here
        self.b,self.erb_order,self.EarQ,self.min_bw=b,erb_order,ear_Q,min_bw
        erb = ((cf/ear_Q)**erb_order + min_bw**erb_order)**(1/erb_order)
        B = b*2*np.pi*erb
#        B = 2*pi*b

        A0 = T
        A2 = 0
        B0 = 1
        B1 = -2*np.cos(2*cf*np.pi*T)/np.exp(B*T)
        B2 = np.exp(-2*B*T)
        
        A11 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) + 2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T) / \

                np.exp(B*T))/2
        A12=-(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T)-2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/\
                np.exp(B*T))/2
        A13=-(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T)+2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/\
                np.exp(B*T))/2
        A14=-(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T)-2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/\
                np.exp(B*T))/2

        i=1j
        gain=abs((-2*np.exp(4*i*cf*np.pi*T)*T+\
                         2*np.exp(-(B*T)+2*i*cf*np.pi*T)*T*\
                                 (np.cos(2*cf*np.pi*T)-np.sqrt(3-2**(3./2))*\
                                  np.sin(2*cf*np.pi*T)))*\
                   (-2*np.exp(4*i*cf*np.pi*T)*T+\
                     2*np.exp(-(B*T)+2*i*cf*np.pi*T)*T*\
                      (np.cos(2*cf*np.pi*T)+np.sqrt(3-2**(3./2))*\
                       np.sin(2*cf*np.pi*T)))*\
                   (-2*np.exp(4*i*cf*np.pi*T)*T+\
                     2*np.exp(-(B*T)+2*i*cf*np.pi*T)*T*\
                      (np.cos(2*cf*np.pi*T)-\
                       np.sqrt(3+2**(3./2))*np.sin(2*cf*np.pi*T)))*\
                   (-2*np.exp(4*i*cf*np.pi*T)*T+2*np.exp(-(B*T)+2*i*cf*np.pi*T)*T*\
                   (np.cos(2*cf*np.pi*T)+np.sqrt(3+2**(3./2))*np.sin(2*cf*np.pi*T)))/\
                  (-2/np.exp(2*B*T)-2*np.exp(4*i*cf*np.pi*T)+\
                   2*(1+np.exp(4*i*cf*np.pi*T))/np.exp(B*T))**4)

        allfilts=np.ones(len(cf))

        self.A0, self.A11, self.A12, self.A13, self.A14, self.A2, self.B0, self.B1, self.B2, self.gain=\
            A0*allfilts, A11, A12, A13, A14, A2*allfilts, B0*allfilts, B1, B2, gain



        self.filt_a=np.dstack((np.array([np.ones(len(cf)), B1, B2]).T,)*4)
        self.filt_b=np.dstack((np.array([A0/gain, A11/gain, A2/gain]).T,
                         np.array([A0*np.ones(len(cf)), A12, np.zeros(len(cf))]).T,
                         np.array([A0*np.ones(len(cf)), A13, np.zeros(len(cf))]).T,
                         np.array([A0*np.ones(len(cf)), A14, np.zeros(len(cf))]).T))

        # The filterbank is a cascade of 4 IIR filters
        f0 = LinearFilterbankGroup(sound, self.filt_b[:,:,0], self.filt_a[:,:,0])
        f1 = LinearFilterbankGroup(f0, self.filt_b[:,:,1], self.filt_a[:,:,1])
        f2 = LinearFilterbankGroup(f1, self.filt_b[:,:,2], self.filt_a[:,:,2])
        f3 = LinearFilterbankGroup(f2, self.filt_b[:,:,3], self.filt_a[:,:,3])
        
        ####################################
        # Brian Group infrastructure stuff #
        ####################################
        # add contained objects
        self._contained_objects += [f0, f1, f2, f3]

        # set up the variables
        self.variables = Variables(self)

        # this line gives the name of the output variable
        self.variables.add_reference('out', f3.variables['out']) # here goes the fancy indexing for Repeat/Tile etc.

        self.variables.add_constant('N', Unit(1), Nchannels) # a group has to have an N
        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be after all variables are set
        self._enable_group_attributes()

    def __len__(self):
        # this should probably not be needed in the future
        return self.N

