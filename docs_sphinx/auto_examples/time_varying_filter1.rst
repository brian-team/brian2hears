.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_time_varying_filter1.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_time_varying_filter1.py:


Time varying filter (1)
-----------------------
This example implements a band pass filter whose center frequency is modulated
by an Ornstein-Uhlenbeck. The white noise term used for this process is output
by a FunctionFilterbank. The bandpass filter coefficients update is an example
of how to use a :class:`~brian2hears.ControlFilterbank`. The bandpass filter is
a basic biquadratic filter for which the Q factor and the center frequency must
be given. The input is a white noise.



.. image:: /auto_examples/images/sphx_glr_time_varying_filter1_001.png
    :alt: time varying filter1
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/marcel/conda/envs/brian2hears/lib/python3.8/site-packages/matplotlib/mlab.py:498: RuntimeWarning: invalid value encountered in multiply
      result = result * window.reshape((-1, 1))
    /home/marcel/conda/envs/brian2hears/lib/python3.8/site-packages/matplotlib/mlab.py:510: RuntimeWarning: overflow encountered in multiply
      result = np.conj(result) * result
    /home/marcel/conda/envs/brian2hears/lib/python3.8/site-packages/matplotlib/mlab.py:510: RuntimeWarning: invalid value encountered in multiply
      result = np.conj(result) * result
    /home/marcel/conda/envs/brian2hears/lib/python3.8/site-packages/matplotlib/mlab.py:532: RuntimeWarning: overflow encountered in multiply
      result[slc] *= scaling_factor
    /home/marcel/conda/envs/brian2hears/lib/python3.8/site-packages/matplotlib/mlab.py:532: RuntimeWarning: invalid value encountered in multiply
      result[slc] *= scaling_factor
    /home/marcel/conda/envs/brian2hears/lib/python3.8/site-packages/matplotlib/mlab.py:538: RuntimeWarning: invalid value encountered in true_divide
      result /= Fs
    /home/marcel/conda/envs/brian2hears/lib/python3.8/site-packages/matplotlib/mlab.py:541: RuntimeWarning: invalid value encountered in true_divide
      result /= (np.abs(window)**2).sum()






|


.. code-block:: default


    from brian2 import *
    from brian2hears import *

    samplerate = 20*kHz
    SoundDuration = 300*ms
    sound = whitenoise(SoundDuration, samplerate).ramp() 

    #number of frequency channel (here it must be one as a spectrogram of the
    #output is plotted)
    nchannels = 1   

    fc_init = 5000*Hz   #initial center frequency of the band pass filter
    Q = 5               #quality factor of the band pass filter
    update_interval = 4 # the filter coefficients are updated every 4 samples

    #parameters of the Ornstein-Uhlenbeck process
    s_i = 1200*Hz
    tau_i = 100*ms      
    mu_i = fc_init/tau_i
    sigma_i = sqrt(2)*s_i/sqrt(tau_i)
    deltaT = defaultclock.dt

    #this function  is used in a FunctionFilterbank. It outputs a noise term that
    #will be later used by the controler to update the center frequency
    noise = lambda x: mu_i*deltaT+sigma_i*randn(1)*sqrt(deltaT)
    noise_generator = FunctionFilterbank(sound, noise)

    #this class will take as input the output of the noise generator and as target
    #the bandpass filter center frequency
    class CoeffController(object):
        def __init__(self, target):
            self.target = target
            self.deltaT = 1./samplerate
            self.BW = 2*arcsinh(1./2/Q)*1.44269
            self.fc = fc_init
        
        def __call__(self, input):
            #the control variables are taken as the last of the buffer
            noise_term = input[-1,:]
            #update the center frequency by updateing the OU process
            self.fc = asarray(self.fc-self.fc/tau_i*self.deltaT)+noise_term

            w0 = 2*pi*self.fc/float(samplerate)
            #update the coefficient of the biquadratic filterbank
            alpha = sin(w0)*sinh(log(2)/2*self.BW*w0/sin(w0))
            self.target.filt_b[:, 0, 0] = sin(w0)/2
            self.target.filt_b[:, 1, 0] = 0
            self.target.filt_b[:, 2, 0] = -sin(w0)/2
    
            self.target.filt_a[:, 0, 0] = 1+alpha
            self.target.filt_a[:, 1, 0] = -2*cos(w0)
            self.target.filt_a[:, 2, 0] = 1-alpha
        
    # In the present example the time varying filter is a LinearFilterbank therefore
    #we must initialise the filter coefficients; the one used for the first buffer computation
    w0 = 2*pi*fc_init/samplerate
    BW = 2*arcsinh(1./2/Q)*1.44269
    alpha = sin(w0)*sinh(log(2)/2*BW*w0/sin(w0))

    filt_b = zeros((nchannels, 3, 1))
    filt_a = zeros((nchannels, 3, 1))
    filt_b[:, 0, 0] = sin(w0)/2
    filt_b[:, 1, 0] = 0
    filt_b[:, 2, 0] = -sin(w0)/2
    filt_a[:, 0, 0] = 1+alpha
    filt_a[:, 1, 0] = -2*cos(w0)
    filt_a[:, 2, 0] = 1-alpha

    #the filter which will have time varying coefficients
    bandpass_filter = LinearFilterbank(sound, filt_b, filt_a)
    #the updater
    updater = CoeffController(bandpass_filter)

    #the controller. Remember it must be the last of the chain
    control = ControlFilterbank(bandpass_filter, noise_generator, bandpass_filter,
                                updater, update_interval)          

    time_varying_filter_mon = control.process()

    figure(1)
    pxx, freqs, bins, im = specgram(squeeze(time_varying_filter_mon),
                                    NFFT=256, Fs=float(samplerate), noverlap=240)
    imshow(flipud(pxx), aspect='auto')

    show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.842 seconds)


.. _sphx_glr_download_auto_examples_time_varying_filter1.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: time_varying_filter1.py <time_varying_filter1.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: time_varying_filter1.ipynb <time_varying_filter1.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
