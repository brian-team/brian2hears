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
s = clicks(1*ms, 5, 2*ms)
plot(s)
show()
