.. currentmodule:: brian2hears

Reference
=========

.. autofunction:: set_default_samplerate

Sounds
------

.. autoclass:: Sound

.. autofunction:: savesound
.. autofunction:: loadsound
.. autofunction:: play(*sounds, normalise=False, sleep=False)
.. autofunction:: whitenoise
.. autofunction:: powerlawnoise
.. autofunction:: brownnoise
.. autofunction:: pinknoise
.. autofunction:: irns
.. autofunction:: irno
.. autofunction:: tone
.. autofunction:: click
.. autofunction:: clicks
.. autofunction:: harmoniccomplex
.. autofunction:: silence
.. autofunction:: sequence(*sounds, samplerate=None)

.. index::
	single: dB
	single: decibel
	pair: level; sound
	single: sound; dB
	single: sound; decibel

.. _dB:

dB
~~

.. autoclass:: dB_type
.. autoclass:: dB_error

Filterbanks
-----------

.. autoclass:: LinearFilterbank
.. autoclass:: FIRFilterbank
.. autoclass:: RestructureFilterbank
.. autoclass:: Join
.. autoclass:: Interleave
.. autoclass:: Repeat
.. autoclass:: Tile
.. autoclass:: FunctionFilterbank
.. autoclass:: SumFilterbank
.. autoclass:: DoNothingFilterbank
.. autoclass:: ControlFilterbank
.. autoclass:: CombinedFilterbank

Filterbank library
------------------
.. autoclass:: Gammatone(source, cf,b=1.019,erb_order=1,ear_Q=9.26449,min_bw=24.7)
.. autoclass:: ApproximateGammatone
.. autoclass:: LogGammachirp(source, f, b=1.019, c=1, ncascades=4)
.. autoclass:: LinearGammachirp
.. autoclass:: LinearGaborchirp
.. autoclass:: IIRFilterbank
.. autoclass:: Butterworth
.. autoclass:: Cascade
.. autoclass:: LowPass
.. autoclass:: AsymmetricCompensation

Auditory model library
----------------------
.. autoclass:: DRNL
.. autoclass:: DCGC
.. autoclass:: MiddleEar
.. autoclass:: TanCarney
.. autoclass:: ZhangSynapse

Filterbank group
----------------

.. autoclass:: FilterbankGroup

Functions
---------

.. autofunction:: erbspace
.. autofunction:: asymmetric_compensation_coeffs

Plotting
--------

.. autofunction:: log_frequency_xaxis_labels
.. autofunction:: log_frequency_yaxis_labels

HRTFs
-----

.. autoclass:: HRTF
.. autoclass:: HRTFSet
.. autoclass:: HRTFDatabase
.. autofunction:: make_coordinates

.. autoclass:: IRCAM_LISTEN
.. autoclass:: HeadlessDatabase

Base classes
------------

Useful for understanding more about the internals.

.. autoclass:: Bufferable

.. autoclass:: Filterbank

.. autoclass:: BaseSound

.. _brian-hears-class-diagram:

Class diagram
-------------

.. inheritance-diagram:: Sound
						 Filterbank
						   LinearFilterbank
						     Gammatone ApproximateGammatone LogGammachirp
						     LinearGammachirp LinearGaborchirp
						   Cascade
						   IIRFilterbank
						     Butterworth
						     LowPass
						   FIRFilterbank
						   RestructureFilterbank
						     Join Interleave Repeat Tile
						   FunctionFilterbank
						     SumFilterbank
						   DoNothingFilterbank
						   ControlFilterbank
						   CombinedFilterbank
						   DRNL DCGC TanCarney
						   AsymmetricCompensation
						 HRTFDatabase
						   IRCAM_LISTEN HeadlessDatabase
	:parts: 1
