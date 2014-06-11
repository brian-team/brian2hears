import time

from brian2 import *
from brian2hears import *

from brian2hears.dev.performance_comparison.benchmark_parameters import parameters

def do_one(Ncf, duration):
    sound = TimedArray(np.random.randn(np.rint(duration/defaultclock.dt)), dt = defaultclock.dt)

    cf = np.array(erbspace(20*Hz, 20000.*Hz, Ncf))
    cochlea = GammatoneFilterbank(sound, cf)

    # Leaky integrate-and-fire model with noise
    eqs = '''
    dv/dt = (I_in-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1
    I_in = 3*clip(I, 0, Inf)**(1./3.) : 1
    '''
    anf = NeuronGroup(len(cf), eqs, reset='v=0', threshold='v>1')
    anf.variables.add_reference('I', cochlea.variables['out'])
    
    run(duration)
    
if __name__ == '__main__':
    Ncfs = parameters['Ncfs']
    durations = parameters['durations']
    fn = parameters['b2h_python_fn']

    results = np.zeros((len(Ncfs), len(durations)))

    print 'Starting tests'
    print 'Ncfs:'
    print Ncfs
    print 'durations:'
    print durations
    print 'Go!'
    for kncf in range(len(Ncfs)):    
        print kncf
        for kduration in range(len(durations)):
            t0 = time.time()
            do_one(Ncfs[kncf],durations[kduration]*second)
            results[kncf, kduration] = time.time()-t0

    np.savez(fn, results)
    print 'All done'


