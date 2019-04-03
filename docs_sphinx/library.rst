.. currentmodule:: brian2hears

Library
-------

Brian hears comes with a package of predefined filter classes to be used as
basic blocks by the user. All of them are implemented as filterbanks.

First, a series of standard filters widely used in audio processing are available:


.. tabularcolumns::|p{3cm}|p{15cm}|p{3cm}|

+------------------------------------+---------------------------------------------------------------------------------------------------+----------------------------------------------+
| Class                              | Descripition                                                                                      |  Example                                     |
+====================================+===================================================================================================+==============================================+
| :class:`IIRFilterbank`             | Bank of low, high, bandpass or bandstop filter of type Chebyshef, Elliptic, etc...                | :ref:`IIR filterbank`                        |
+------------------------------------+---------------------------------------------------------------------------------------------------+----------------------------------------------+
| :class:`Butterworth`               | Bank of low, high, bandpass or bandstop Butterworth filters                                       | :ref:`Butterworth filters`                   |
+------------------------------------+---------------------------------------------------------------------------------------------------+----------------------------------------------+
| :class:`LowPass`                   | Bank of lowpass filters of order 1                                                                | :ref:`Cochleagram`                           |
+------------------------------------+---------------------------------------------------------------------------------------------------+----------------------------------------------+

Second, the library provides linear auditory filters developed to model the
middle ear transfer function and the frequency analysis of the cochlea:

.. tabularcolumns::|p{3cm}|p{15cm}|p{3cm}|

+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| Class                              | Description                                                                                       |  Example                                                   |
+====================================+===================================================================================================+============================================================+
| :class:`MiddleEar`                 | Linear bandpass filter, based on middle-ear frequency response properties                         | :ref:`Spiking output of the Tan&Carney model`              |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| :class:`Gammatone`                 | Bank of IIR gammatone filters  (based on Slaney implementation)                                   | :ref:`Gammatone filters`                                   |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| :class:`ApproximateGammatone`      | Bank of IIR gammatone filters  (based on Hohmann implementation)                                  | :ref:`Approximate Gammatone filters`                       |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| :class:`LogGammachirp`             | Bank of IIR gammachirp filters with logarithmic sweep (based on Irino implementation)             | :ref:`Logarithmic Gammachirp filters`                      |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| :class:`LinearGammachirp`          | Bank of FIR chirp filters with linear sweep and gamma envelope                                    | :ref:`Linear Gammachirp filters`                           |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| :class:`LinearGaborchirp`          | Bank of FIR chirp filters with linear sweep and gaussian envelope                                 |                                                            |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+

Finally, Brian hears comes with a series of complex nonlinear cochlear models
developed to model nonlinear effects such as filter bandwith level dependency,
two-tones suppression, peak position level dependency, etc.

.. tabularcolumns::|p{3cm}|p{15cm}|p{3cm}|

+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| Class                              | Description                                                                                       |  Example                                                   |
+====================================+===================================================================================================+============================================================+
| :class:`DRNL`                      | Dual resonance nonlinear filter as described in Lopez-Paveda and Meddis, JASA 2001                | :ref:`Dual resonance nonlinear filter (DRNL)`              |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| :class:`DCGC`                      | Compressive gammachirp auditory filter as described in  Irino and Patterson, JASA 2001            | :ref:`Compressive Gammachirp filter (DCGC)`                |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| :class:`TanCarney`                 | Auditory phenomenological model as described in  Tan and Carney, JASA 2003                        | :ref:`Spiking output of the Tan&Carney model`              |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| :class:`ZhangSynapse`              | Model of an inner hair cell -- auditory nerve synapse (Zhang et al., JASA 2001)                   | :ref:`Spiking output of the Tan&Carney model`              |
+------------------------------------+---------------------------------------------------------------------------------------------------+------------------------------------------------------------+
