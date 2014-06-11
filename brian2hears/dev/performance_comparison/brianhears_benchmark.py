import time

from brian import *
from brian.hears import *

from brian2hears.dev.performance_comparison.benchmark_parameters import parameters

def do_one(Ncf, duration):
    sound = whitenoise(duration)
    
    cf = erbspace(20*Hz, 20*kHz, Ncf)
    cochlea = Gammatone(sound, cf)
    # Half-wave rectification and compression [x]^(1/3)
    ihc = FunctionFilterbank(cochlea, lambda x: 3*clip(x, 0, Inf)**(1.0/3.0))
    # Leaky integrate-and-fire model with noise
    eqs = '''
    dv/dt = (I-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1
    I : 1
    '''
    anf = FilterbankGroup(ihc, 'I', eqs, reset=0, threshold=1)#, refractory=5*ms)
    run(sound.duration)

if __name__ == '__main__':
    Ncfs = parameters['Ncfs']
    durations = parameters['durations']
    fn = parameters['bh_python_fn']

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
                   
