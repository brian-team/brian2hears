Plotting
--------

Often, you want to use log-scaled axes for frequency in plots, but the
built-in matplotlib axis labelling for log-scaled axes doesn't work well for
frequencies. We provided two functions (:func:`log_frequency_xaxis_labels` and
:func:`log_frequency_yaxis_labels`) to automatically set useful axis labels.
For example::

	cf = erbspace(100*Hz, 10*kHz)
	...
	semilogx(cf, response)
	axis('tight')
	log_frequency_xaxis_labels()

