from brian2 import *
from numpy import *
import numpy
import array as pyarray
import time
import struct
pygame_loaded = False
try:
    from scikits.samplerate import resample
    have_scikits_samplerate = True
except (ImportError, ValueError):
    have_scikits_samplerate = False
from bufferable import Bufferable
from prefs import get_samplerate
from db import dB, dB_type, dB_error, gain
from scipy.signal import fftconvolve, lfilter
from scipy.misc import factorial

__all__ = ['BaseSound', 'Sound',
           'pinknoise','brownnoise','powerlawnoise',
           'whitenoise', 'irns', 'irno', 
           'tone', 'click', 'clicks', 'silence', 'sequence', 'harmoniccomplex',
           'loadsound', 'savesound', 'play', 'vowel'
           ]


_mixer_status = [-1,-1]

class BaseSound(Bufferable):
    '''
    Base class for Sound and OnlineSound
    '''
    pass

class Sound(BaseSound, numpy.ndarray):
    '''
    Class for working with sounds, including loading/saving, manipulating and playing.

    For an overview, see :ref:`sounds_overview`.
    
    **Initialisation**
    
    The following arguments are used to initialise a sound object
    
    ``data``
        Can be a filename, an array, a function or a sequence (list or tuple).
        If its a filename, the sound file (WAV or AIFF) will be loaded. If its
        an array, it should have shape ``(nsamples, nchannels)``. If its a
        function, it should be a function f(t). If its a sequence, the items
        in the sequence can be filenames, functions, arrays or Sound objects.
        The output will be a multi-channel sound with channels the corresponding
        sound for each element of the sequence. 
    ``samplerate=None``
        The samplerate, if necessary, will use the default (for an array or
        function) or the samplerate of the data (for a filename).
    ``duration=None``
        The duration of the sound, if initialising with a function.
  
    **Loading, saving and playing**
    
    .. automethod:: load
    .. automethod:: save
    .. automethod:: play

    **Properties**
    
    .. autoattribute:: duration
    .. autoattribute:: nsamples
    .. autoattribute:: nchannels
    .. autoattribute:: times
    .. autoattribute:: left
    .. autoattribute:: right
    .. automethod:: channel
    
    **Generating sounds**
    
    All sound generating methods can be used with durations arguments in samples (int) or units (e.g. 500*ms). One can also set the number of channels by setting the keyword argument nchannels to the desired value. Notice that for noise the channels will be generated independantly.
    
    .. automethod:: tone
    .. automethod:: whitenoise
    .. automethod:: powerlawnoise
    .. automethod:: brownnoise
    .. automethod:: pinknoise
    .. automethod:: silence
    .. automethod:: click
    .. automethod:: clicks
    .. automethod:: harmoniccomplex
    .. automethod:: vowel
    
    **Timing and sequencing**
    
    .. automethod:: sequence(*sounds, samplerate=None)
    .. automethod:: repeat
    .. automethod:: extended
    .. automethod:: shifted
    .. automethod:: resized

    **Slicing**
    

    One can slice sound objects in various ways, for example ``sound[100*ms:200*ms]``
    returns the part of the sound between 100 ms and 200 ms (not including the
    right hand end point). If the sound is less than 200 ms long it will be
    zero padded. You can also set values using slicing, e.g.
    ``sound[:50*ms] = 0`` will silence the first 50 ms of the sound. The syntax
    is the same as usual for Python slicing. In addition, you can select a
    subset of the channels by doing, for example, ``sound[:, -5:]`` would be
    the last 5 channels. For time indices, either times or samples can be given,
    e.g. ``sound[:100]`` gives the first 100 samples. In addition, steps can
    be used for example to reverse a sound as ``sound[::-1]``.
    
    **Arithmetic operations**
    
    Standard arithemetical operations and numpy functions work as you would
    expect with sounds, e.g. ``sound1+sound2``, ``3*sound`` or ``abs(sound)``.
    
    **Level**
    
    .. autoattribute:: level
    .. automethod:: atlevel
    .. autoattribute:: maxlevel
    .. automethod:: atmaxlevel
    
    **Ramping**
    
    .. automethod:: ramp
    .. automethod:: ramped
    
    **Plotting**
    
    .. automethod:: spectrogram
    .. automethod:: spectrum
    '''
    duration = property(fget=lambda self:len(self) / self.samplerate,
                        doc='The length of the sound in seconds.')
    nsamples = property(fget=lambda self:len(self),
                        doc='The number of samples in the sound.')
    times = property(fget=lambda self:arange(len(self), dtype=float) / self.samplerate,
                     doc='An array of times (in seconds) corresponding to each sample.')
    nchannels = property(fget=lambda self:self.shape[1],
                         doc='The number of channels in the sound.')
    left = property(fget=lambda self:self.channel(0),
                    doc='The left channel for a stereo sound.')
    right = property(fget=lambda self:self.channel(1),
                     doc='The right channel for a stereo sound.')

    @check_units(samplerate=Hz, duration=second)
    def __new__(cls, data, samplerate=None, duration=None):
        if isinstance(data, numpy.ndarray):
            samplerate = get_samplerate(samplerate)
#            if samplerate is None:
#                raise ValueError('Must specify samplerate to initialise Sound with array.')
            if duration is not None:
                raise ValueError('Cannot specify duration when initialising Sound with array.')
            x = array(data, dtype=float)
        elif isinstance(data, str):
            if duration is not None:
                raise ValueError('Cannot specify duration when initialising Sound from file.')
            if samplerate is not None:
                raise ValueError('Cannot specify samplerate when initialising Sound from a file.')
            x = Sound.load(data)
            samplerate = x.samplerate
        elif callable(data):
            samplerate = get_samplerate(samplerate)
#            if samplerate is None:
#                raise ValueError('Must specify samplerate to initialise Sound with function.')
            if duration is None:
                raise ValueError('Must specify duration to initialise Sound with function.')
            L = int(rint(duration * samplerate))
            t = arange(L, dtype=float) / samplerate
            x = data(t)
        elif isinstance(data, (list, tuple)):
            kwds = {}
            if samplerate is not None:
                kwds['samplerate'] = samplerate
            if duration is not None:
                kwds['duration'] = duration
            channels = tuple(Sound(c, **kwds) for c in data)
            x = hstack(channels)
            samplerate = channels[0].samplerate
        else:
            raise TypeError('Cannot initialise Sound with data of class ' + str(data.__class__))
        if len(x.shape)==1:
            x.shape = (len(x), 1)
        x = x.view(cls)
        x.samplerate = samplerate
        x.buffer_init()
        return x

    def __array_wrap__(self, obj, context=None):
        handled = False
        x = numpy.ndarray.__array_wrap__(self, obj, context)
        if not hasattr(x, 'samplerate') and hasattr(self, 'samplerate'):
            x.samplerate = self.samplerate
        if context is not None:
            ufunc = context[0]
            args = context[1]
        return x
    
    def __array_finalize__(self,obj):
        if obj is None: return
        self.samplerate = getattr(obj, 'samplerate', None)
        
    def buffer_init(self):
        pass
        
    def buffer_fetch(self, start, end):
        if start<0:
            raise IndexError('Can only use positive indices in buffer.')
        samples = end-start
        X = asarray(self)[start:end, :]
        if X.shape[0]<samples:
            X = vstack((X, zeros((samples-X.shape[0], X.shape[1]))))
        return X

    def channel(self, n):
        '''
        Returns the nth channel of the sound.
        '''
        return Sound(self[:, n], self.samplerate)

    def __add__(self, other):
        if isinstance(other, Sound):
            if int(other.samplerate) > int(self.samplerate):
                self = self.resample(other.samplerate)
            elif int(other.samplerate) < int(self.samplerate):
                other = other.resample(self.samplerate)

            if len(self) > len(other):
                other = other.resized(len(self))
            elif len(self) < len(other):
                self = self.resized(len(other))

            return Sound(numpy.ndarray.__add__(self, other), samplerate=self.samplerate)
        else:
            x = numpy.ndarray.__add__(self, other)
            return Sound(x, self.samplerate)
    __radd__ = __add__

    def __getitem__(self, key):
        channel = slice(None)
        if isinstance(key, tuple):
            channel = key[1]
            key = key[0]

        if isinstance(key, int):
            return np.ndarray.__getitem__(self, key)
        if isinstance(key, float):
            return np.ndarray.__getitem__(self, round(key*self.samplerate))

        sliceattr = [v for v in [key.start, key.stop] if v is not None]
        slicedims = array([units.have_same_dimensions(flag, second) for flag in sliceattr])
        attrisint = array([isinstance(v, int) for v in sliceattr])
        s = sum(attrisint)
        if s!=0 and s!=len(sliceattr):
            raise ValueError('Slice attributes must be all ints or all times')
        if s==len(sliceattr): # all ints
            start = key.start or 0
            stop = key.stop or self.shape[0]
            step = key.step or 1
            if start>=0 and stop<=self.shape[0]:
                return Sound(np.ndarray.__getitem__(self, (key, channel)),
                             self.samplerate)
            else:
                startpad = max(-start, 0)
                endpad = max(stop-self.shape[0], 0)
                startmid = max(start, 0)
                endmid = min(stop, self.shape[0])
                atstart = zeros((startpad, self.shape[1]))
                atend = zeros((endpad, self.shape[1]))
                return Sound(vstack((atstart,
                                     asarray(self)[startmid:endmid:step],
                                     atend)), self.samplerate)
        if not slicedims.all():
            raise DimensionMismatchError('Slicing',
                                        *[units.get_unit(d) for d in sliceattr])
        
        start = key.start or 0*msecond
        stop = key.stop or self.duration
        step = key.step or 1
        if int(step)!=step:
            #resampling
            raise NotImplementedError
        start = int(rint(start*self.samplerate))
        stop = int(rint(stop*self.samplerate))
        return self.__getitem__((slice(start,stop,step),channel))
    
    def __setitem__(self,key,value):
        channel=slice(None)
        if isinstance(key,tuple):
            channel=key[1]
            key=key[0]
        
        if isinstance(key,int):
            return np.ndarray.__setitem__(self,(key,channel),value)
        if isinstance(key,float):
            return np.ndarray.__setitem__(self,(int(rint(key*self.samplerate)),channel),value)

        sliceattr = [v for v in [key.start, key.step, key.stop] if v is not None]
        slicedims = array([units.have_same_dimensions(flag, second) for flag in sliceattr])
        attrisint = array([isinstance(v, int) for v in sliceattr])
        s = sum(attrisint)
        if s!=0 and s!=len(sliceattr):
            raise ValueError('Slice attributes must be all ints or all times')
        if s==len(sliceattr): # all ints
            # If value is a mono sound its shape will be (N, 1) but the numpy
            # setitem will have shape (N,) so in this case it's a shape mismatch
            # so we squeeze the array to make sure this doesn't happen.
            if isinstance(value,Sound) and channel!=slice(None):
                value=value.squeeze()
            return asarray(self).__setitem__((key,channel),value) # returns None

        if not slicedims.all():
            raise DimensionMismatchError('Slicing',*[units.get_unit(d) for d in sliceattr])

        if key.__getattribute__('step') is not None:
            # resampling?
            raise NotImplementedError
        
        start = key.start
        stop = key.stop or self.duration
        if (start is not None and start<0*ms) or stop > self.duration:
            raise IndexError('Slice bigger than Sound object')
        if start is not None: start = int(rint(start*self.samplerate))
        stop = int(rint(stop*self.samplerate))
        return self.__setitem__((slice(start,stop),channel),value)

    def extended(self, duration):
        '''
        Returns the Sound with length extended by the given duration, which
        can be the number of samples or a length of time in seconds.
        '''
        duration = get_duration(duration, self.samplerate)
        return self[:self.nsamples+duration]

    def resized(self, L):
        '''
        Returns the Sound with length extended (or contracted) to have L samples.
        '''
        if L == len(self):
            return self
        elif L < len(self):
            return Sound(self[:L, :], samplerate=self.samplerate)
        else:
            padding = zeros((L - len(self), self.nchannels))
            return Sound(concatenate((self, padding)), samplerate=self.samplerate)

    def shifted(self, duration, fractional=False, filter_length=2048):
        '''
        Returns the sound delayed by duration, which can be the number of
        samples or a length of time in seconds. Normally, only integer
        numbers of samples will be used, but if ``fractional=True`` then
        the filtering method from
        `http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html <http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html>`__
        will be used (introducing some small numerical errors). With this
        method, you can specify the ``filter_length``, larger values are
        slower but more accurate, especially at higher frequencies. The large
        default value of 2048 samples provides good accuracy for sounds with
        frequencies above 20 Hz, but not for lower frequency sounds. If you are
        restricted to high frequency sounds, a smaller value will be more
        efficient. Note that if ``fractional=True`` then
        ``duration`` is assumed to be a time not a number of samples.
        '''
        if not fractional:
            if not isinstance(duration, int):
                duration = int(rint(duration*self.samplerate))
            if duration>=0:
                y = vstack((zeros((duration, self.nchannels)), self))
                return Sound(y, samplerate=self.samplerate)
            else:
                return self[-duration:, :]
        else:
            if self.nchannels>1:
                sounds = [self.channel(i).shifted(duration, fractional=True, filter_length=filter_length) for i in xrange(self.nchannels)]
                return Sound(hstack(sounds), samplerate=self.samplerate)
            # Adapted from
            # http://www.labbookpages.co.uk/audio/beamforming/fractionalDelay.html
            delay = duration*self.samplerate
            if delay>=0:
                idelay = int(delay)
            elif delay<0:
                idelay = -int(-delay)
            delay -= idelay
            centre_tap = filter_length // 2
            t = arange(filter_length)
            x = t-delay
            if abs(round(delay)-delay)<1e-10:
                tap_weight = array(x==centre_tap, dtype=float)
            else:
                sinc = sin(pi*(x-centre_tap))/(pi*(x-centre_tap))
                window = 0.54-0.46*cos(2.0*pi*(x+0.5)/filter_length) # Hamming window
                tap_weight = window*sinc
            if filter_length<256:
                y = convolve(tap_weight, self.flatten())
            else:
                y = fftconvolve(tap_weight, self.flatten())
            y = y[filter_length/2:-filter_length/2]
            sound = Sound(y, self.samplerate)
            sound = sound.shifted(idelay)
            return sound

    def repeat(self, n):
        '''
        Repeats the sound n times
        '''
        x = vstack((self,)*n)
        return Sound(x, samplerate=self.samplerate)

    ### TODO: test this - I haven't installed scikits.samplerate on windows
    # it should work, according to the documentation 2D arrays are acceptable
    # in the format we use fof sounds here
    @check_units(samplerate=Hz)
    def resample(self, samplerate, resample_type='sinc_best'):
        '''
        Returns a resampled version of the sound.
        '''
        if not have_scikits_samplerate:
            raise ImportError('Need scikits.samplerate package for resampling')
        y = array(resample(self, float(samplerate / self.samplerate), resample_type),
                  dtype=float64)
        return Sound(y, samplerate=samplerate)

    def _init_mixer(self):
        global _mixer_status
        if _mixer_status==[-1,-1] or _mixer_status[0]!= self.nchannels or _mixer_status[1] != self.samplerate:
            pygame.mixer.quit()
            pygame.mixer.init(int(self.samplerate), -16, self.nchannels)
            _mixer_status=[self.nchannels,self.samplerate]

    
    def play(self, normalise=False, sleep=False):
        '''
        Plays the sound (normalised to avoid clipping if required). If
        sleep=True then the function will wait until the sound has finished
        playing before returning.
        '''
        global pygame, pygame_loaded
        if not pygame_loaded:
            try:
                import pygame
            except ImportError:
                raise ImportError("Playing sounds requires the pygame module to be installed")
            pygame_loaded = True
        if self.nchannels>2:
            raise ValueError("Can only play sounds with 1 or 2 channels.")
        self._init_mixer()
        if normalise:
            a = amax(abs(self))
        else:
            a = 1
        x = array((2 ** 15 - 1) * clip(self / a, -1, 1), dtype=int16)
        if self.nchannels==1:
            x.shape = x.size
        # Make sure pygame receives an array in C-order        
        x = pygame.sndarray.make_sound(ascontiguousarray(x))
        x.play()
        if sleep:
            time.sleep(self.duration)

    def spectrogram(self, low=None, high=None, log_power=True, other = None,  **kwds):
        '''
        Plots a spectrogram of the sound
        
        Arguments:
        
        ``low=None``, ``high=None``
            If these are left unspecified, it shows the full spectrogram,
            otherwise it shows only between ``low`` and ``high`` in Hz.
        ``log_power=True``
            If True the colour represents the log of the power.
        ``**kwds``
            Are passed to Pylab's ``specgram`` command.
        
        Returns the values returned by pylab's ``specgram``, namely
        ``(pxx, freqs, bins, im)`` where ``pxx`` is a 2D array of powers,
        ``freqs`` is the corresponding frequencies, ``bins`` are the time bins,
        and ``im`` is the image axis.
        '''
        if self.nchannels>1:
            raise ValueError('Can only plot spectrograms for mono sounds.')
        if other is not None:
            x = self.flatten()-other.flatten()
        else:
            x = self.flatten()
        pxx, freqs, bins, im = specgram(x, Fs=self.samplerate, **kwds)
        if low is not None or high is not None:
            restricted = True
            if low is None:
                low = 0*Hz
            if high is None:
                high = amax(freqs)*Hz
            I = logical_and(low <= freqs, freqs <= high)
            I2 = where(I)[0]
            I2 = [max(min(I2) - 1, 0), min(max(I2) + 1, len(freqs) - 1)]
            Z = pxx[I2[0]:I2[-1], :]
        else:
            restricted = False
            Z = pxx
        if log_power:
            Z[Z < 1e-20] = 1e-20 # no zeros because we take logs
            Z = 10 * log10(Z)
        Z = flipud(Z)
        if restricted:
            imshow(Z, extent=(0, amax(bins), freqs[I2[0]], freqs[I2[-1]]),
                   origin='upper', aspect='auto')
        else:
            imshow(Z, extent=(0, amax(bins), freqs[0], freqs[-1]),
                   origin='upper', aspect='auto')
        xlabel('Time (s)')
        ylabel('Frequency (Hz)')
        return (pxx, freqs, bins, im)

    @check_units(low=Hz, high=Hz)
    def spectrum(self, low=None, high=None, log_power=True, display=False):
        '''
        Returns the spectrum of the sound and optionally plots it.
        
        Arguments:
        
        ``low``, ``high`` 
            If these are left unspecified, it shows the full spectrum,
            otherwise it shows only between ``low`` and ``high`` in Hz.
        ``log_power=True``
            If True it returns the log of the power.
        ``display=False``
            Whether to plot the output.
        
        Returns ``(Z, freqs, phase)``
        where ``Z`` is a 1D array of powers, ``freqs`` is the corresponding
        frequencies, ``phase`` is the unwrapped phase of spectrum.
        '''
        if self.nchannels>1:
            raise ValueError('Can only plot spectrum for mono sounds.')

        # Flatten array, fft operates on the last axis by default
        sp = numpy.fft.fft(array(self).flatten())
        freqs = array(range(len(sp)), dtype=float64) / len(sp) * float64(self.samplerate)
        pxx = abs(sp) ** 2
        phase = unwrap(mod(angle(sp), 2 * pi))
        if low is not None or high is not None:
            restricted = True
            if low is None:
                low = 0*Hz
            if high is None:
                high = amax(freqs)*Hz
            I = logical_and(low <= freqs, freqs <= high)
            I2 = where(I)[0]
            Z = pxx[I2]
            freqs = freqs[I2]
            phase = phase[I2]
        else:
            restricted = False
            Z = pxx
        if log_power:
            Z[Z < 1e-20] = 1e-20 # no zeros because we take logs
            Z = 10 * log10(Z)
        if display:
            subplot(211)
            semilogx(freqs, Z)
            ticks_freqs = 32000 * 2 ** -array(range(18), dtype=float64)
            xticks(ticks_freqs, map(str, ticks_freqs))
            grid()
            xlim((freqs[0], freqs[-1]))
            xlabel('Frequency (Hz)')
            ylabel('Power (dB/Hz)') if log_power else ylabel('Power')
            subplot(212)
            semilogx(freqs, phase)
            ticks_freqs = 32000 * 2 ** -array(range(18), dtype=float64)
            xticks(ticks_freqs, map(str, ticks_freqs))
            grid()
            xlim((freqs[0], freqs[-1]))
            xlabel('Frequency (Hz)')
            ylabel('Phase (rad)')
            show()
        return (Z, freqs, phase)

    def get_level(self):
        '''
        Returns level in dB SPL (RMS) assuming array is in Pascals.
        In the case of multi-channel sounds, returns an array of levels
        for each channel, otherwise returns a float.
        '''
        if self.nchannels==1:
            rms_value = sqrt(mean((asarray(self)-mean(asarray(self)))**2))
            rms_dB = 20.0*log10(rms_value/2e-5)
            return rms_dB*dB
        else:
            return array(tuple(self.channel(i).get_level() for i in xrange(self.nchannels)))

    def set_level(self, level):
        '''
        Sets level in dB SPL (RMS) assuming array is in Pascals. ``level``
        should be a value in dB, or a tuple of levels, one for each channel.
        '''
        rms_dB = self.get_level()
        if self.nchannels>1:
            level = array(level)
            if level.size==1:
                level = level.repeat(self.nchannels)
            level = reshape(level, (1, self.nchannels))
            rms_dB = reshape(rms_dB, (1, self.nchannels))
        else:
            if not isinstance(level, dB_type):
                raise dB_error('Must specify level in dB')
            rms_dB = float(rms_dB)
            level = float(level)
        gain = 10**((level-rms_dB)/20.)
        self *= gain

    level = property(fget=get_level, fset=set_level, doc='''
        Can be used to get or set the level of a sound, which should be in dB.
        For single channel sounds a value in dB is used, for multiple channel
        sounds a value in dB can be used for setting the level (all channels
        will be set to the same level), or a list/tuple/array of levels. It
        is assumed that the unit of the sound is Pascals.
        ''')
    
    def atlevel(self, level):
        '''
        Returns the sound at the given level in dB SPL (RMS) assuming array is
        in Pascals. ``level`` should be a value in dB, or a tuple of levels,
        one for each channel.
        '''
        newsound = self.copy()
        newsound.level = level
        return newsound
    
    def get_maxlevel(self):
        return amax(self.level)*dB
    
    def set_maxlevel(self, level):
        self.level += level-self.maxlevel
        
    maxlevel = property(fget=get_maxlevel, fset=set_maxlevel, doc='''
        Can be used to set or get the maximum level of a sound. For mono
        sounds, this is the same as the level, but for multichannel sounds
        it is the maximum level across the channels. Relative level differences
        will be preserved. The specified level should be a value in dB, and it
        is assumed that the unit of the sound is Pascals. 
        ''')

    def atmaxlevel(self, level):
        '''
        Returns the sound with the maximum level across channels set to the
        given level. Relative level differences will be preserved. The specified
        level should be a value in dB and it is assumed that the unit of the
        sound is Pascals.
        '''
        newsound = self.copy()
        newsound.maxlevel = level
        return newsound
            
    def ramp(self, when='onset', duration=10*ms, envelope=None, inplace=True):
        '''
        Adds a ramp on/off to the sound
        
        ``when='onset'``
            Can take values 'onset', 'offset' or 'both'
        ``duration=10*ms``
            The time over which the ramping happens
        ``envelope``
            A ramping function, if not specified uses ``sin(pi*t/2)**2``. The
            function should be a function of one variable ``t`` ranging from
            0 to 1, and should increase from ``f(0)=0`` to ``f(0)=1``. The
            reverse is applied for the offset ramp.
        ``inplace``
            Whether to apply ramping to current sound or return a new array.
        '''
        when = when.lower().strip()
        if envelope is None: envelope = lambda t:sin(pi * t / 2) ** 2
        if not isinstance(duration, int):
            sz = int(rint(duration * self.samplerate))
        else:
            sz = duration
        multiplier = envelope(reshape(linspace(0.0, 1.0, sz), (sz, 1)))
        if inplace:
            target = self
        else:
            target = Sound(copy(self), self.samplerate)
        if when == 'onset' or when == 'both':
            target[:sz, :] *= multiplier
        if when == 'offset' or when == 'both':
            target[target.nsamples-sz:, :] *= multiplier[::-1]
        return target
    
    @check_units(duration=second)
    def ramped(self, when='onset', duration=10*ms, envelope=None):
        '''
        Returns a ramped version of the sound (see :meth:`Sound.ramp`).
        '''
        return self.ramp(when=when, duration=duration, envelope=envelope, inplace=False)

    def fft(self,n=None):
        '''
        Performs an n-point FFT on the sound object, that is an array of the same size containing the DFT of each channel. n defaults to the number of samples of the sound, but can be changed manually setting the ``n`` keyword argument
        '''
        if n is None:
            n=self.shape[0]
        res=zeros(n,self.nchannels)
        for i in range(self.nchannels):
            res[:,i]=fft(asarray(self)[:,i].flatten(),n=n)
        return res
            
        
    
    @staticmethod
    @check_units(frequency=Hz, duration=second, samplerate=Hz)
    def tone(frequency, duration, phase=0, samplerate=None, nchannels=1):
        '''
        Returns a pure tone at frequency for duration, using the default
        samplerate or the given one. The ``frequency`` and ``phase`` parameters
        can be single values, in which case multiple channels can be
        specified with the ``nchannels`` argument, or they can be sequences
        (lists/tuples/arrays) in which case there is one frequency or phase for
        each channel.
        '''
        samplerate = get_samplerate(samplerate)
        duration = get_duration(duration, samplerate)
        frequency = asarray(frequency)*Hz
        phase = array(phase)
        if frequency.size>nchannels and nchannels==1:
            nchannels = frequency.size
        if phase.size>nchannels and nchannels==1:
            nchannels = phase.size
        if frequency.size==nchannels:
            frequency.shape = (1, nchannels)
        if phase.size==nchannels:
            phase.shape =(nchannels, 1)
        t = arange(0, duration, 1)/samplerate
        t.shape = (t.size, 1) # ensures C-order (in contrast to tile(...).T )
        x = sin(phase + 2.0 * pi * frequency * tile(t, (1, nchannels)))
        return Sound(x, samplerate)

    @staticmethod
    @check_units(f0=Hz, duration=second, samplerate=Hz)
    def harmoniccomplex(f0, duration, amplitude=1, phase=0, samplerate=None, nchannels=1):
        '''
        Returns a harmonic complex composed of pure tones at integer multiples
        of the fundamental frequency ``f0``. 
        The ``amplitude`` and
        ``phase`` keywords can be set to either a single value or an
        array of values. In the former case the value is set for all
        harmonics, and harmonics up to the sampling frequency are
        generated. In the latter each harmonic parameter is set
        separately, and the number of harmonics generated corresponds
        to the length of the array.
        '''
        samplerate=get_samplerate(samplerate)

        phases = np.array(phase).flatten()
        amplitudes = np.array(amplitude).flatten()
        
        if len(phases)>1 or len(amplitudes)>1:
            if (len(phases)>1 and len(amplitudes)>1) and (len(phases) != len(amplitudes)):
                raise ValueError('Please specify the same number of phases and amplitudes')        
            Nharmonics = max(len(phases),len(amplitudes)) 
        else:
            Nharmonics = int(np.floor( samplerate/(2*f0) ) )
            
        if len(phases) == 1:
            phases = np.tile(phase, Nharmonics)
        if len(amplitudes) == 1:
            amplitudes = np.tile(amplitude, Nharmonics)
            
        x = amplitudes[0]*tone(f0, duration, phase = phases[0], 
                               samplerate = samplerate, nchannels = nchannels)
        for i in range(1,Nharmonics):
            x += amplitudes[i]*tone((i+1)*f0, duration, phase = phases[i], 
                                    samplerate = samplerate, nchannels = nchannels)
        return Sound(x,samplerate)
    
    @staticmethod
    @check_units(duration=second, samplerate=Hz)
    def whitenoise(duration, samplerate=None, nchannels=1):
        '''
        Returns a white noise. If the samplerate is not specified, the global
        default value will be used.
        '''
        samplerate = get_samplerate(samplerate)
        duration = get_duration(duration,samplerate)
        x = randn(duration,nchannels)
        return Sound(x, samplerate)

    @staticmethod
    @check_units(duration=second, samplerate=Hz)
    def powerlawnoise(duration, alpha, samplerate=None, nchannels=1,normalise=False):
        '''
        Returns a power-law noise for the given duration. Spectral density per unit of bandwidth scales as 1/(f**alpha).
        
        Sample usage::
        
            noise = powerlawnoise(200*ms, 1, samplerate=44100*Hz)
        
        Arguments:
        
        ``duration`` 
            Duration of the desired output.
        ``alpha``
            Power law exponent.
        ``samplerate``
            Desired output samplerate
        '''
        samplerate = get_samplerate(samplerate)
        duration = get_duration(duration,samplerate)
        
        # Adapted from http://www.eng.ox.ac.uk/samp/software/powernoise/powernoise.m
        # Little MA et al. (2007), "Exploiting nonlinear recurrence and fractal
        # scaling properties for voice disorder detection", Biomed Eng Online, 6:23
        n=duration
        n2=int(n/2)
        
        f=array(fftfreq(n,d=1.0/samplerate), dtype=complex)
        f.shape=(len(f),1)
        f=tile(f,(1,nchannels))
        
        if n%2==1:
            z=(randn(n2,nchannels)+1j*randn(n2,nchannels))
            a2=1.0/( f[1:(n2+1),:]**(alpha/2.0))
        else:
            z=(randn(n2-1,nchannels)+1j*randn(n2-1,nchannels))
            a2=1.0/(f[1:n2,:]**(alpha/2.0))
        
        a2*=z
        
        if n%2==1:
            d=vstack((ones((1,nchannels)),a2,
                      flipud(conj(a2))))
        else:
            d=vstack((ones((1,nchannels)),a2,
                      1.0/( abs(f[n2])**(alpha/2.0) )*
                      randn(1,nchannels),
                      flipud(conj(a2))))
        
        
        x=real(ifft(d.flatten()))                  
        x.shape=(n,nchannels)

        if normalise:
            for i in range(nchannels):
                #x[:,i]=normalise_rms(x[:,i])
                x[:,i] = ((x[:,i] - amin(x[:,i]))/(amax(x[:,i]) - amin(x[:,i])) - 0.5) * 2;
        
        return Sound(x,samplerate)
    
            
    @staticmethod
    @check_units(duration=second, samplerate=Hz)
    def pinknoise(duration, samplerate=None, nchannels=1, normalise=False):
        '''
        Returns pink noise, i.e :func:`powerlawnoise` with alpha=1
        '''
        return Sound.powerlawnoise(duration, 1.0, samplerate=samplerate,
                                   nchannels=nchannels, normalise=normalise)
    
    @staticmethod
    @check_units(duration=second, samplerate=Hz)
    def brownnoise(duration, samplerate=None, nchannels=1, normalise=False):
        '''
        Returns brown noise, i.e :func:`powerlawnoise` with alpha=2
        '''
        return Sound.powerlawnoise(duration, 2.0, samplerate=samplerate,
                                   nchannels=nchannels, normalise=normalise)
    
    @staticmethod
    @check_units(duration=second, samplerate=Hz)
    def irns(delay, gain, niter, duration, samplerate=None, nchannels=1):
        '''
        Returns an IRN_S noise. The iterated ripple noise is obtained trough
        a cascade of gain and delay filtering. 
        For more details: see Yost 1996 or chapter 15 in Hartman Sound Signal Sensation.
        '''
        if nchannels!=1:
            raise ValueError("nchannels!=1 not supported.")
        samplerate = get_samplerate(samplerate)
        noise=Sound.whitenoise(duration)
        splrate=noise.samplerate
        x=array(noise.T)[0]
        IRNfft=np.fft.fft(x)
        Nspl,spl_dur=len(IRNfft),float(1.0/splrate)
        w=2*pi*fftfreq(Nspl,spl_dur)
        d=float(delay)
        for k in range(1,niter+1):
            nchoosek=factorial(niter)/(factorial(niter-k)*factorial(k))
            IRNfft+=nchoosek*(gain**k)*IRNfft*exp(-1j*w*k*d)
        IRNadd = np.fft.ifft(IRNfft)
        x=real(IRNadd)
        return Sound(x,samplerate)
    
    @staticmethod
    @check_units(duration=second, samplerate=Hz)
    def irno(delay, gain, niter, duration, samplerate=None, nchannels=1):
        '''
        Returns an IRN_O noise. The iterated ripple noise is obtained many attenuated and
        delayed version of the original broadband noise. 
        For more details: see Yost 1996 or chapter 15 in Hartman Sound Signal Sensation.
        '''
        samplerate = get_samplerate(samplerate)
        noise=Sound.whitenoise(duration)
        splrate=noise.samplerate
        x=array(noise.T)[0]
        IRNadd=np.fft.fft(x)
        Nspl,spl_dur=len(IRNadd),float(1.0/splrate)
        w=2*pi*fftfreq(Nspl,spl_dur)
        d=float(delay)
        for k in range(1,niter+1):
            IRNadd+=(gain**k)*IRNadd*exp(-1j*w*k*d)
        IRNadd = np.fft.ifft(IRNadd)
        x=real(IRNadd)
        return Sound(x, samplerate)

    @staticmethod
    @check_units(duration=second, samplerate=Hz)
    def click(duration, peak=None, samplerate=None, nchannels=1):
        '''
        Returns a click of the given duration.
        
        If ``peak`` is not specified, the amplitude will be 1, otherwise
        ``peak`` refers to the peak dB SPL of the click, according to the
        formula ``28e-6*10**(peak/20.)``.
        '''
        samplerate = get_samplerate(samplerate)
        duration = get_duration(duration,samplerate)
        if peak is not None:
            if not isinstance(peak, dB_type):
                raise dB_error('Peak must be given in dB')
            amplitude = 28e-6*10**(float(peak)/20.)
        else:
            amplitude = 1
        x = amplitude*ones((duration,nchannels))
        return Sound(x, samplerate)
    
    @staticmethod
    @check_units(duration=second, samplerate=Hz)
    def clicks(duration, n, interval, peak=None, samplerate=None, nchannels=1):
        '''
        Returns a series of n clicks (see :func:`click`) separated by interval.
        '''
        oneclick = Sound.click(duration, peak=peak, samplerate=samplerate)
        return oneclick[slice(None, interval)].repeat(n)

    @staticmethod
    @check_units(duration=second, samplerate=Hz)
    def silence(duration, samplerate=None, nchannels=1):
        '''
        Returns a silent, zero sound for the given duration. Set nchannels to set the number of channels.
        '''
        samplerate = get_samplerate(samplerate)
        duration = get_duration(duration,samplerate)
        x=numpy.zeros((duration,nchannels))
        return Sound(x, samplerate)

    @staticmethod
    @check_units(pitch=Hz, duration=second, samplerate=Hz)
    def vowel(vowel=None, formants=None, pitch=100*Hz, duration=1*second,
          samplerate=None, nchannels=1):
        '''
        Returns an artifically created spoken vowel sound (following the 
        source-filter model of speech production) with a given ``pitch``.
        
        The vowel can be specified by either providing ``vowel`` as a string
        ('a', 'i' or 'u') or by setting ``formants`` to a sequence of formant
        frequencies.
        
        The returned sound is normalized to a maximum amplitude of 1.
        
        The implementation is based on the MakeVowel function written by Richard
        O. Duda, part of the Auditory Toolbox for Matlab by Malcolm Slaney:
        http://cobweb.ecn.purdue.edu/~malcolm/interval/1998-010/                
        '''    
        
        samplerate = get_samplerate(samplerate)
        duration = get_duration(duration, samplerate)
        
        if not (vowel or formants):
            raise ValueError('Need either a vowel or a list of formants')
        elif (vowel and formants):
            raise ValueError('Cannot use both vowel and formants')
            
        if vowel:
            if vowel == 'a' or vowel == '/a/':
                formants = (730.0*Hz, 1090.0*Hz, 2440.0*Hz)
            elif vowel == 'i' or vowel == '/i/':
                formants = (270.0*Hz, 2290.0*Hz, 3010.0*Hz)
            elif vowel == 'u' or vowel == '/u/':
                formants = (300.0*Hz, 870.0*Hz, 2240.0*Hz)
            else:
                raise ValueError('Unknown vowel: "%s"' % (vowel))            
        
        points = np.arange(0, duration - 1, samplerate / pitch)
            
        indices = np.floor(points).astype(int)
        
        y = np.zeros(duration)
    
        y[indices] = (indices + 1) - points
        y[indices + 1] = points - indices
        
        # model the sound source (periodic glottal excitation)  
        a = np.exp(-250.*Hz * 2 * np.pi / samplerate)
        y = lfilter([1],[1, 0, -a * a], y.copy())
        
        # model the filtering by the vocal tract
        bandwidth = 50.*Hz
        
        for f in formants:
            cft = f / samplerate
            q = f / bandwidth
            rho = np.exp(-np.pi * cft / q)
            theta = 2 * np.pi * cft * np.sqrt(1 - 1/(4.0 * q * q))
            a2 = -2 * rho * np.cos(theta)
            a3 = rho * rho
            y = lfilter([1 + a2 + a3], [1, a2, a3], y.copy()) 
        
        #normalize sound
        data = y / np.max(np.abs(y), axis=0)        
        data.shape = (data.size, 1)
        return Sound(np.tile(data, (nchannels, 1)),  samplerate=samplerate)

    @staticmethod
    def sequence(*args, **kwds):
        '''
        Returns the sequence of sounds in the list sounds joined together
        '''
        samplerate = kwds.pop('samplerate', None)
        if len(kwds):
            raise TypeError('Unexpected keywords to function sequence()')
        sounds = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                sounds.extend(arg)
            else:
                sounds.append(arg)
        if samplerate is None:
            samplerate = max(s.samplerate for s in sounds)
            rates = unique([int(s.samplerate) for s in sounds])
            if len(rates)>1:
                sounds = tuple(s.resample(samplerate) for s in sounds)
        x = vstack(sounds)
        return Sound(x, samplerate)

    def save(self, filename, normalise=False, samplewidth=2):
        '''
        Save the sound as a WAV.
        
        If the normalise keyword is set to True, the amplitude of the sound will be
        normalised to 1. The samplewidth keyword can be 1 or 2 to save the data as
        8 or 16 bit samples.
        '''
        ext = filename.split('.')[-1].lower()
        if ext=='wav':
            import wave as sndmodule
        elif ext=='aiff' or ext=='aifc':
            import aifc as sndmodule
            raise NotImplementedError('Can only save as wav soundfiles')
        else:
            raise NotImplementedError('Can only save as wav soundfiles')
        
        if samplewidth != 1 and samplewidth != 2:
            raise ValueError('Sample width must be 1 or 2 bytes.')
        
        scale = {2:2**15-1, 1: 2**7-1}[samplewidth]
        if ext=='wav':
            meanval = {2:0, 1:2**7}[samplewidth]
            dtype = {2:int16, 1:uint8}[samplewidth]
            typecode = {2:'h', 1:'B'}[samplewidth]
        else:
            meanval = {2:0, 1:2**7}[samplewidth]
            dtype = {2:int16, 1:uint8}[samplewidth]
            typecode = {2:'h', 1:'B'}[samplewidth]
        w = sndmodule.open(filename, 'wb')
        w.setnchannels(self.nchannels)
        w.setsampwidth(samplewidth)
        w.setframerate(int(self.samplerate))
        x = array(self,copy=True)
        am=amax(x)
        z = zeros(x.shape[0]*self.nchannels, dtype=x.dtype)
        x.shape=(x.shape[0],self.nchannels)
        for i in range(self.nchannels):
            if normalise:
                x[:,i] /= am
            x[:,i] = (x[:,i]) * scale + meanval
            z[i::self.nchannels] = x[::1,i]
        data = array(z, dtype=dtype)
        data = pyarray.array(typecode, data)
        w.writeframes(data.tostring())
        w.close()
    
    @staticmethod
    def load(filename):
        '''
        Load the file given by filename and returns a Sound object. 
        Sound file can be either a .wav or a .aif file.
        '''
        ext = filename.split('.')[-1].lower()
        if ext=='wav':
            import wave as sndmodule
        elif ext=='aif' or ext=='aiff':
            import aifc as sndmodule
        else:
            raise NotImplementedError('Can only load aif or wav soundfiles')
        wav = sndmodule.open(filename, "r")
        nchannels, sampwidth, framerate, nframes, comptype, compname = wav.getparams()
        frames = wav.readframes(nframes * nchannels)
        typecode = {2:'h', 1:'B'}[sampwidth]
        out = frombuffer(frames, dtype=dtype(typecode))
        scale = {2:2 ** 15, 1:2 ** 7-1}[sampwidth]
        meanval = {2:0, 1:2**7}[sampwidth]
        
        data = zeros((nframes, nchannels))
        for i in range(nchannels):
            data[:, i] = out[i::nchannels]
            data[:, i] /= scale
            data[:, i] -= meanval
        
        return Sound(data, samplerate=framerate*Hz)

    def __repr__(self):
        arrayrep = repr(asarray(self))
        arrayrep = '\n'.join('    '+l for l in arrayrep.split('\n'))
        return 'Sound(\n'+arrayrep+',\n    '+repr(self.samplerate)+')'
    
    def __str__(self):
        return 'Sound duration %s, channels %s, samplerate %s' % (self.duration,
                                                                  self.nchannels,
                                                                  self.samplerate)

    def __reduce__(self):
        return (_load_Sound_from_pickle, (asarray(self), float(self.samplerate)))


def _load_Sound_from_pickle(arr, samplerate):
    return Sound(arr, samplerate=samplerate*Hz)

def play(*sounds, **kwds):
    '''
    Plays a sound or sequence of sounds. For example::
    
        play(sound)
        play(sound1, sound2)
        play([sound1, sound2, sound3])
        
    If ``normalise=True``, the sequence of sounds will be normalised to the
    maximum range (-1 to 1), and if ``sleep=True`` the function will wait
    until the sounds have finished playing before returning.
    '''
    normalise = kwds.pop('normalise', False)
    sleep = kwds.pop('sleep', False)
    if len(kwds):
        raise TypeError('Unexpected keyword arguments to function play()')
    sound = sequence(*sounds)
    sound.play(normalise=normalise, sleep=sleep)
play.__doc__ = Sound.play.__doc__

def savesound(sound, filename, normalise=False, samplewidth=2):
    sound.save(filename, normalise=normalise, samplewidth=samplewidth)
savesound.__doc__ = Sound.save.__doc__

def get_duration(duration,samplerate):
    if not isinstance(duration, int):
        duration = int(rint(duration * samplerate))
    return duration


whitenoise = Sound.whitenoise
powerlawnoise = Sound.powerlawnoise
pinknoise = Sound.pinknoise
brownnoise = Sound.brownnoise
irns = Sound.irns
irno = Sound.irno
tone = Sound.tone
harmoniccomplex = Sound.harmoniccomplex
click = Sound.click
clicks = Sound.clicks
silence = Sound.silence
sequence = Sound.sequence
vowel = Sound.vowel
loadsound = Sound.load
