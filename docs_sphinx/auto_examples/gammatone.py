#!/usr/bin/env python
'''
Gammatone filters
-----------------
Example of the use of the class :class:`~brian2hears.Gammatone` available in the
library. It implements a fitlerbank of IIR gammatone filters as 
described  in Slaney, M., 1993, "An Efficient Implementation of the
Patterson-Holdsworth Auditory Filter Bank". Apple Computer Technical Report #35. 
In this example, a white noise is filtered by a gammatone filterbank and the
resulting cochleogram is plotted.
'''
from brian2 import *
from brian2hears import *
from matplotlib import pyplot

sound = whitenoise(100*ms).ramp()
sound.level = 50*dB

nbr_center_frequencies = 50
b1 = 1.019  #factor determining the time constant of the filters
#center frequencies with a spacing following an ERB scale
center_frequencies = erbspace(100*Hz, 1000*Hz, nbr_center_frequencies)
gammatone = Gammatone(sound, center_frequencies, b=b1)

gt_mon = gammatone.process()

figure()
imshow(gt_mon.T, aspect='auto', origin='lower',
       extent=(0, sound.duration/ms,
               center_frequencies[0]/Hz, center_frequencies[-1]/Hz))
pyplot.yscale('log')
title('Cochleogram')
ylabel('Frequency (Hz)')
xlabel('Time (ms)')

show()
