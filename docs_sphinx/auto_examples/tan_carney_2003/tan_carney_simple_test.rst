.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_tan_carney_2003_tan_carney_simple_test.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_tan_carney_2003_tan_carney_simple_test.py:


Spiking output of the Tan&Carney model
--------------------------------------
Fig. 1 and 3 (spking output without spiking/refractory period) should
reproduce the output of the AN3_test_tone.m and AN3_test_click.m
scripts, available in the code accompanying the paper Tan & Carney (2003).
This matlab code is available from
http://www.urmc.rochester.edu/labs/Carney-Lab/publications/auditory-models.cfm

Tan, Q., and L. H. Carney.
"A Phenomenological Model for the Responses of Auditory-nerve Fibers.
II. Nonlinear Tuning with a Frequency Glide".
The Journal of the Acoustical Society of America 114 (2003): 2007.



.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/tan_carney_2003/images/sphx_glr_tan_carney_simple_test_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/tan_carney_2003/images/sphx_glr_tan_carney_simple_test_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/tan_carney_2003/images/sphx_glr_tan_carney_simple_test_003.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/tan_carney_2003/images/sphx_glr_tan_carney_simple_test_004.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Testing click response
    Testing tone response




|


.. code-block:: default


    import numpy as np
    import matplotlib.pyplot as plt

    from brian2 import *
    from brian2hears import (Sound, get_samplerate, set_default_samplerate, tone,
                             click, silence, dB, TanCarney, MiddleEar, ZhangSynapse)

    set_default_samplerate(50*kHz)
    sample_length = 1 / get_samplerate(None)
    cf = 1000 * Hz

    print('Testing click response')
    duration = 25*ms    
    levels = [40, 60, 80, 100, 120]
    # a click of two samples
    tones = Sound([Sound.sequence([click(sample_length*2, peak=level*dB),
                                   silence(duration=duration - sample_length)])
               for level in levels])
    ihc = TanCarney(MiddleEar(tones), [cf] * len(levels), update_interval=1)
    syn = ZhangSynapse(ihc, cf)
    mon = StateMonitor(syn, ['s', 'R'], record=True, clock=syn.clock)
    spike_mon = SpikeMonitor(syn)
    net = Network(syn, mon, spike_mon)
    net.run(duration * 1.5)

    spiketimes = spike_mon.spike_trains()

    for idx, level in enumerate(levels):
        plt.figure(1)
        plt.subplot(len(levels), 1, idx + 1)
        plt.plot(mon.t/ms, mon.s[idx])
        plt.xlim(0, 25)
        plt.xlabel('Time (msec)')
        plt.ylabel('Sp/sec')
        plt.text(15, np.nanmax(mon.s[idx])/2., 'Peak SPL=%s SPL' % str(level*dB));
        ymin, ymax = plt.ylim()
        if idx == 0:
            plt.title('Click responses')

        plt.figure(2)
        plt.subplot(len(levels), 1, idx + 1)
        plt.plot(mon.t/ms, mon.R[idx])
        plt.xlabel('Time (msec)')
        plt.xlabel('Time (msec)')
        plt.text(15, np.nanmax(mon.s[idx])/2., 'Peak SPL=%s SPL' % str(level*dB));
        plt.ylim(ymin, ymax)
        if idx == 0:
            plt.title('Click responses (with spikes and refractoriness)')
        plt.plot(spiketimes[idx]/ms,
             np.ones(len(spiketimes[idx])) * np.nanmax(mon.R[idx]), 'rx')

    print('Testing tone response')
    duration = 60*ms    
    levels = [0, 20, 40, 60, 80]
    tones = Sound([Sound.sequence([tone(cf, duration).atlevel(level*dB).ramp(when='both',
                                                                             duration=10*ms,
                                                                             inplace=False),
                                   silence(duration=duration/2)])
                   for level in levels])
    ihc = TanCarney(MiddleEar(tones), [cf] * len(levels), update_interval=1)
    syn = ZhangSynapse(ihc, cf)
    mon = StateMonitor(syn, ['s', 'R'], record=True, clock=syn.clock)
    spike_mon = SpikeMonitor(syn)
    net = Network(syn, mon, spike_mon)
    net.run(duration * 1.5)

    spiketimes = spike_mon.spike_trains()

    for idx, level in enumerate(levels):
        plt.figure(3)
        plt.subplot(len(levels), 1, idx + 1)
        plt.plot(mon.t/ms, mon.s[idx])
        plt.xlim(0, 120)
        plt.xlabel('Time (msec)')
        plt.ylabel('Sp/sec')
        plt.text(1.25 * duration/ms, np.nanmax(mon.s[idx])/2., '%s SPL' % str(level*dB));
        ymin, ymax = plt.ylim()
        if idx == 0:
            plt.title('CF=%.0f Hz - Response to Tone at CF' % cf)

        plt.figure(4)
        plt.subplot(len(levels), 1, idx + 1)
        plt.plot(mon.t/ms, mon.R[idx])
        plt.xlabel('Time (msec)')
        plt.xlabel('Time (msec)')
        plt.text(1.25 * duration/ms, np.nanmax(mon.R[idx])/2., '%s SPL' % str(level*dB));
        plt.ylim(ymin, ymax)
        if idx == 0:
            plt.title('CF=%.0f Hz - Response to Tone at CF (with spikes and refractoriness)' % cf)
        plt.plot(spiketimes[idx] / ms,
             np.ones(len(spiketimes[idx])) * np.nanmax(mon.R[idx]), 'rx')

    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  29.187 seconds)


.. _sphx_glr_download_auto_examples_tan_carney_2003_tan_carney_simple_test.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tan_carney_simple_test.py <tan_carney_simple_test.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tan_carney_simple_test.ipynb <tan_carney_simple_test.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
