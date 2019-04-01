.. index::
    single: sound
    pair: sound; wav
    pair: sound; aiff

.. _sounds_overview:

Sounds
------

Sounds can be loaded from a WAV or AIFF file with the :func:`loadsound`
function (and saved with the :func:`savesound` function or :meth:`Sound.save`
method), or by initialising with a filename::

    sound = loadsound('test.wav')
    sound = Sound('test.aif')
    sound.save('test.wav')

Various standard types of sounds can also be constructed, e.g. pure tones,
white noise, clicks and silence::

    sound = tone(1*kHz, 1*second)
    sound = whitenoise(1*second)
    sound = click(1*ms)
    sound = silence(1*second)

You can pass a function of time or an array to initialise a sound::

    # Equivalent to Sound.tone
    sound = Sound(lambda t:sin(50*Hz*2*pi*t), duration=1*second)

    # Equivalent to Sound.whitenoise
    sound = Sound(randn(int(1*second*44.1*kHz)), samplerate=44.1*kHz)

Multiple channel sounds can be passed as a list or tuple of filenames,
arrays or :class:`Sound` objects::

    sound = Sound(('left.wav', 'right.wav'))
    sound = Sound((randn(44100), randn(44100)), samplerate=44.1*kHz)
    sound = Sound((Sound.tone(1*kHz, 1*second),
                   Sound.tone(2*kHz, 1*second)))

A multi-channel sound is also a numpy array of shape ``(nsamples, nchannels)``,
and can be initialised as this (or converted to a standard numpy array)::

    sound = Sound(randn(44100, 2), samplerate=44.1*kHz)
    arr = array(sound)

Sounds can be added and multiplied::

    sound = Sound.tone(1*kHz, 1*second)+0.1*Sound.whitenoise(1*second)

For more details on combining and operating on sounds, including shifting them
in time, repeating them, resampling them, ramping them, finding and setting
intensities, plotting spectrograms, etc., see :class:`Sound`.

Sounds can be played using the :func:`play` function or :meth:`Sound.play` method::

    play(sound)
    sound.play()

.. index::
    pair: sound; sequence

Sequences of sounds can be played as::

    play(sound1, sound2, sound3)

.. index::
    pair: sound; stereo
    single: sound; multiple channels

The number of channels in a sound can be found using the ``nchannels``
attribute, and individual channels can be extracted using the
:meth:`Sound.channel` method, or using the ``left`` and ``right`` attributes
in the case of stereo sounds::

    print sound.nchannels
    print amax(abs(sound.left-sound.channel(0)))

As an example of using this, the following swaps the channels in a stereo sound::

    sound = Sound('test_stereo.wav')
    swappedsound = Sound((sound.right, sound.left))
    swappedsound.play()

.. index::
    pair: sound; level
    pair: sound; dB

The level of the sound can be computed and changed with the ``sound.level``
attribute. Levels are returned in dB which is a special unit in Brian hears.
For example, ``10*dB+10`` will raise an error because ``10`` does not have
units of dB. The multiplicative gain of a value in dB can be computed with
the function ``gain(level)``. All dB values are measured as RMS dB SPL assuming
that the values of the sound object are measured in Pascals. Some examples::

    sound = whitenoise(100*ms)
    print sound.level
    sound.level = 60*dB
    sound.level += 10*dB
    sound *= gain(-10*dB)

