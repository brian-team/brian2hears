.. currentmodule:: brian2hears

.. index::
    pair: buffering; interface

Buffering interface
-------------------

The :class:`Sound` and :class:`Filterbank` classes
(and all classes derived from them) all implement the same buffering
mechanism. The purpose of this is to allow for efficient processing of
multiple channels in buffers. Rather than precomputing the application of
filters to all channels (which for large numbers of channels or long sounds
would not fit in memory), we process small chunks at a time. The entire design
of these classes is based on the idea of buffering, as defined by the base
class :class:`Bufferable` (see section :ref:`brian-hears-class-diagram`).
Each class
has two methods, ``buffer_init()`` to initialise the buffer, and
``buffer_fetch(start, end)`` to fetch the portion of the buffer from samples
with indices from ``start`` to ``end`` (not including ``end`` as standard for
Python). The ``buffer_fetch(start, end)`` method should return a 2D array of
shape ``(end-start, nchannels)`` with the buffered values.

From the user point of view, all you need to do, having set up a chain of
:class:`Sound` and :class:`Filterbank` objects, is to call ``buffer_fetch(start, end)``
repeatedly. If the output of a :class:`Filterbank` is being plugged into a
:class:`FilterbankGroup` object, everything is handled automatically. For cases
where the number of channels is small or the length of the input source is short,
you can use the :meth:`Filterbank.process` method to automatically
handle the initialisation and repeated application of ``buffer_fetch``.

To extend :class:`Filterbank`, it is often sufficient just to implement the
``buffer_apply(input)`` method. See the documentation for :class:`Filterbank`
for more details.
