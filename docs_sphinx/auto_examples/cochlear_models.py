#!/usr/bin/env python
'''
Cochlear models
---------------
Example of the use of the cochlear models (:class:`~brian2hears.DRNL`,
:class:`~brian2hears.DCGC` and :class:`~brian2hears.TanCarney`) available in the library.
'''
from brian2 import *
from brian2hears import *

simulation_duration = 50*ms
set_default_samplerate(50*kHz)
sound = whitenoise(simulation_duration)
sound = sound.atlevel(50*dB) # level in rms dB SPL
cf = erbspace(100*Hz, 1000*Hz, 50) # centre frequencies

param_drnl = {}
param_drnl['lp_nl_cutoff_m'] = 1.1

param_dcgc = {}
param_dcgc['c1'] = -2.96

figure(figsize=(10, 4))
for i, (model, param) in enumerate([(DRNL, param_drnl),
                                    (DCGC, param_dcgc),
                                    (TanCarney, None)]):
    fb = model(sound, cf, param=param)
    out = fb.process()
    subplot(1, 3, i+1)
    title(model.__name__)
    imshow(flipud(out.T), aspect='auto', extent=(0, simulation_duration/ms, 0, len(cf)-1))
    xlabel('Time (ms)')
    if i==0:
        ylabel('CF (kHz)')
        yticks([0, len(cf)-1], [cf[0]/kHz, cf[-1]/kHz])
    else:
        yticks([])

tight_layout()
show()
