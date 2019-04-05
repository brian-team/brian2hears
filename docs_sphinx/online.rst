.. currentmodule:: brian2hears

.. index::
    pair: online; computation

Online computation
------------------

Typically in auditory modelling, we precompute the entire output of each
channel of the filterbank ("offline computation"), and then work with that.
This is straightforward,
but puts a severe limit on the number of channels we can use or the length of
time we can work with (otherwise the RAM would be quickly exhausted).
Brian hears allows us to use a very large number of channels in filterbanks,
but at the cost of only storing the output of the filterbanks for a relatively
short period of time ("online computation").
This requires a slight change in the way we use the
output of the filterbanks, but is actually not too difficult. For example,
suppose we wanted to compute the vector of RMS values for each channel of the
output of the filterbank. Traditionally, or if we just use the syntax
``output = fb.process()`` in Brian hears, we have an array ``output`` of
shape ``(nsamples, nchannels)``. We could compute the vector of RMS values as::

    rms = sqrt(mean(output**2, axis=0))

To do the same thing with online computation, we simply store a vector of the
running sum of squares, and update it for each buffered segment as it is
computed. At the end of the processing, we divide the sum of squares by the
number of samples and take the square root.

The :meth:`Filterbank.process`
method allows us to pass an optional function ``f(output, running)`` of
two arguments. In this case, :meth:`~Filterbank.process` will first call
``running = f(output, 0)`` for the first buffered segment ``output``. It will
then call ``running = f(output, running)`` for each subsequent segment. In
other words, it will "accumulate" the output of ``f``, passing the output of
each call to the subsequent call. To compute the vector of RMS values then,
we simply do::

    def sum_of_squares(input, running):
        return running+sum(input**2, axis=0)

    rms = sqrt(fb.process(sum_of_squares)/nsamples)

If the computation you wish to perform is more complicated than can be
achieved with the :meth:`~Filterbank.process` method, you can derive a class
from :class:`Filterbank` (see that class' reference documentation for more
details on this).

