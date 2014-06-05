from brian2.core.variables import Variables
from brian2 import *

class FIRFilterbankGroup(Group):
    '''
    Initialized with the FIR filter coeficients of shape (Nchannels, Ntaps)
    '''
    add_to_magic_network = True
    invalidates_magic_network = True
    def __init__(self, sound, fir_coeficients, 
                 codeobj_class = None, when = None, name = 'firfilterbankgroup*'):
        BrianObject.__init__(self, when = when, name = name)

        # FIR filtering specific stuff
        Nchannels, Ntaps = fir_coeficients.shape[0], fir_coeficients.shape[1]
        self._input = sound

        fir_coeficients = np.zeros((Nchannels, Ntaps), dtype = double)
        for k in range(Nchannels):
            fir_coeficients[k, 64+k] = 1.

        # First, prepare the sr_group that holds the last Ntaps values of the sound
        sr_equations = '''
        x : 1
        x_coef : 1
        shift_indices : integer (constant)
        coeficient : 1 (constant)
        '''
        sr_g = NeuronGroup(Ntaps, model = sr_equations, 
                           codeobj_class = codeobj_class, clock = self.clock)
        sr_g.coeficient = fir_coeficients

        sr_S = Synapses(sr_g, sr_g, 
                        codeobj_class = codeobj_class, clock = self.clock)
        sr_S.connect(Ntaps - 1, Ntaps - 1)#Ntaps-1,Ntaps-1)
        sr_custom_code = sr_S.runner('x_post = sound(t+Ntaps*dt)', 
                                     when = (self.clock, 'start', 1))

        sr_g.shift_indices = np.roll(np.arange(Ntaps, dtype = int), -1)
        sr_g.variables.add_reference('shifted', sr_g.variables['x'], index = 'shift_indices')
        sr_g_custom_code = sr_g.runner('x = shifted', 
                                       when = (self.clock, 'start', 1))

        # Then prepare a synapses group for filtering
        out_g = NeuronGroup(Nchannels, 'filtered : 1', 
                            codeobj_class = codeobj_class, clock = self.clock)
        fir_S = Synapses(sr_g, out_g, '''coef : 1
                                        filtered_post = x_pre*coef : 1 (summed)''', 
                         codeobj_class = codeobj_class, clock = self.clock)

        for k in range(Nchannels):
            fir_S.connect(np.arange(Ntaps), k)

        fir_S.coef = fir_coeficients[:,::-1].flatten()

        # # # # # #
        self._contained_objects += [sr_g, out_g, sr_S, fir_S, sr_g_custom_code, sr_custom_code]

        # set up the variables
        self.variables = Variables(self)
        self.variables.add_reference('out', out_g.variables['filtered']) # here goes the fancy indexing for Repeat/Tile etc.
        self.variables.add_constant('N', Unit(1), Nchannels) # a group has to have an N
        self.variables.add_clock_variables(self.clock)

        # creates natural naming scheme for attributes
        # has to be at the end
        self._enable_group_attributes()

    def __len__(self):
        return self.N



if __name__ == '__main__':
    Nchannels = 100
    Ntaps = 1024
    
    sound = TimedArray(np.random.randn(10000), dt=defaultclock.dt)
    
    coeficients = np.zeros((Nchannels, Ntaps), dtype = double)
    for k in range(Nchannels):
        coeficients[k, 64+k] = 1.
        
    g = FIRFilterbankGroup(sound, coeficients)
    print "o"
    M = StateMonitor(g, 'out', record = True)
    
    net = Network(g, M) 
    net.run(100*ms)
    
    imshow(M.out_)
    show()
        

    
