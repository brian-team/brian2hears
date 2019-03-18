from brian2 import *
from brian2hears import *

#sound = whitenoise(100*ms)
sound = tone(100*Hz, 100*ms)
#fb = FunctionFilterbank(sound, lambda x: abs(x))
fb = Gammatone(sound, erbspace(50*Hz, 200*Hz, 4))
res = fb.process()
plot(sound.times/ms, res)
show()
