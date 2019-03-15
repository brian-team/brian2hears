from brian2 import *

class T(object):
    def __getslice__(self, start, stop):
        print('here')
        return self.__getitem__(slice(start, stop))
    def __getitem__(self, key):
        print('here3')
        if isinstance(key, slice):
            return (key.start, key.stop)

t = T()
# this always works on py2+3 <- probably use this solution unless there's a hack?
print(t[slice(0.3*ms, 0.5*ms)])
# this works on py3
# py2: this works if you remove __getslice__ but otherwise not - problem is numpy.ndarray defines __getslice__
print(t[0.3*ms:0.5*ms])
