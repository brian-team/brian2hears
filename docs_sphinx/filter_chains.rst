.. currentmodule:: brian2hears

.. index::
    single: filter
    single: filter bank
    pair: cochlea; modelling

Filter chains
-------------

The standard way to set up a model based on filterbanks is to start with a
sound and then construct a chain of filterbanks that modify it, for example
a common model of cochlear filtering is to apply a bank of gammatone filters,
and then half wave rectify and compress it (for example, with a 1/3 power law).
This can be achieved in Brian hears as follows (for 3000 channels in the
human hearing range from 20 Hz to 20 kHz)::

    cfmin, cfmax, cfN = 20*Hz, 20*kHz, 3000
    cf = erbspace(cfmin, cfmax, cfN)
    sound = Sound('test.wav')
    gfb = GammatoneFilterbank(sound, cf)
    ihc = FunctionFilterbank(gfb, lambda x: clip(x, 0, Inf)**(1.0/3.0))

The :func:`erbspace` function constructs an array of centre frequencies on the
ERB scale. The ``GammatoneFilterbank(source, cf)`` class creates a bank
of gammatone filters with inputs coming from ``source`` and the centre
frequencies in the array ``cf``. The ``FunctionFilterbank(source, func)``
creates a bank of filters that applies the given function ``func`` to the inputs
in ``source``.

Filterbanks can be added and multiplied, for example for creating a linear and
nonlinear path, e.g.::

    sum_path_fb = 0.1*linear_path_fb+0.2*nonlinear_path_fb

A filterbank must have an input with either a single channel or an equal number
of channels. In the former case, the single channel is duplicated for each of
the output channels. However, you might want to apply gammatone filters to a
stereo sound, for example, but in this case it's not clear how to duplicate
the channels and you have to specify it explicitly. You can do this using the
:class:`Repeat`, :class:`Tile`, :class:`Join` and :class:`Interleave`
filterbanks. For example, if the input is a stereo sound
with channels LR then you can get an output with channels LLLRRR or LRLRLR
by writing (respectively)::

    fb = Repeat(sound, 3)
    fb = Tile(sound, 3)

To combine multiple filterbanks into one, you can either
join them in series or interleave them, as follows::

    fb = Join(source1, source2)
    fb = Interleave(source1, source2)

For a more general (but more complicated) approach, see
:class:`RestructureFilterbank`.

Two of the most important generic filterbanks (upon which many of the others
are based) are :class:`LinearFilterbank` and :class:`FIRFilterbank`. The former
is a generic digital filter for FIR and IIR filters. The latter is specifically
for FIR filters. These can be implemented with the former, but the
implementation is optimised using FFTs with the latter (which can often be
hundreds of times faster, particularly for long impulse responses). IIR filter
banks can be designed using :class:`IIRFilterbank` which is based on the
syntax of the ``iirdesign`` scipy function.

You can change the input source to a :class:`Filterbank` by modifying its
``source`` attribute, e.g. to change the input sound of a filterbank ``fb``
you might do::

    fb.source = newsound

Note that the new source should have the same number of channels.

.. index::
    pair: filtering; control path

You can implement control paths (using the output of one filter chain path
to modify the parameters of another filter chain path) using
:class:`ControlFilterbank` (see reference documentation for more details).
For examples of this in action, see the following:

 * :ref:`Time varying filter (1)`.
 * :ref:`Time varying filter (2)`.
 * :ref:`Compressive Gammachirp filter (DCGC)`.
