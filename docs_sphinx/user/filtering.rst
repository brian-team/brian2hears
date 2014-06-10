Filtering in brian2hears
========================

The basic functionality of brian2hears is to allow one to use linear filters in Brian2 neurons.
As per usually, this may be achieved through two different methods: Infinite Impulse Response (IIR) filtering and FInite Impulse Response filtering.
Brian2hears implements both those strategies in two different objects.


Infinite Impulse Response (IIR) Filtering
-----------------------------------------

.. autoclass:: brian2hears.core.linearfilterbank.LinearFilterbankGroup

Finite Impulse Response (FIR) Filtering
---------------------------------------

.. autoclass:: brian2hears.core.linearfilterbank.FIRFilterbankGroup

Miscellaneous
-------------

.. autoclass:: brian2hears.core.linearfilterbank.ShiftRegisterGroup
