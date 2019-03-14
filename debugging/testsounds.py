from brian2 import *
from brian2hears.sounds import *
import numpy

# s = tone([1]*kHz, 10*ms)
# s = harmoniccomplex(1*kHz, 10*ms, phase=[0, 0, 0])
# s = whitenoise(10*ms)
# s = powerlawnoise(10*ms, 2.0)
# s = pinknoise(10*ms)
# s = brownnoise(10*ms)
# s = irns(1*ms, 0.9, 5, 10*ms)
# s = irno(1*ms, 0.9, 5, 10*ms)
# s = click(10*ms)
#s = clicks(1*ms, 5, 2*ms)
#plot(s)
#show()

class T(object):
    def __getslice__(self, start, stop):
        print 'here'
        return self.__getitem__(slice(start, stop))
    def __getitem__(self, key):
        print 'here3'
        if isinstance(key, slice):
            return (key.start, key.stop)

t = T()
print t[0.3*ms:0.5*ms] # this works if you remove __getslice__ but otherwise not - problem is numpy.ndarray defines __getslice__ - maybe not an issue in Python 3 which has no getslice?
#print t[slice(0.3*ms, 0.5*ms)] # this always works <- probably use this solution unless there's a hack?
