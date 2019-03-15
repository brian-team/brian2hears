from brian2 import *
from brian2hears import *

# s = tone([1, 2]*kHz, 10*ms)
# s = harmoniccomplex(1*kHz, 10*ms, phase=[0, 0, 0])
# s = whitenoise(10*ms)
# s = powerlawnoise(10*ms, 2.0)
# s = pinknoise(10*ms)
# s = brownnoise(10*ms)
# s = irns(1*ms, 0.9, 5, 10*ms)
# s = irno(1*ms, 0.9, 5, 10*ms)
# s = click(10*ms)
# s = clicks(1*ms, 5, 2*ms)
# s = silence(10*ms)
# s2 = vowel('a', duration=100*ms)
#s = sequence([s1, s2])
# s = s1+s2
# s.repeat(10).play(normalise=True)
# s.save('test.wav', normalise=True)
# s = loadsound('test.wav')
# s = Sound('test.wav')
# s = asarray(s)
# print s.__class__
# s = Sound(s)
# print repr(s)
# print str(s)

s = Sound(lambda t: sin(2*pi*1000*Hz*t), duration=20*ms)
s = s.ramp()
# s[slice(100*ms, 200*ms)] = 0
#s = s.shifted(0.5/(44.1*kHz))

# plot(s)
# #plot(s.shifted(0.5/(44.1*kHz), fractional=True))
# #print s.spectrum(display=True)
# print s.level
# plot(s)
# s.level = s.level-5*dB
plot(s.times/ms, s)
# print s.level
show()
