Porting from brian.hears
========================


Here I list the tasks that are left.

Porting Sound objects
---------------------

Essentially ND timedarray. 

Porting parts of filterbank structuring options
-----------------------------------------------

Maybe not all of them. I think that many are too complex. 

Porting the library filterbanks
-------------------------------
This should be relatively straightfoward, provided that the structure of the filterbankgroup matches that of brian.hears

Faster implementation
---------------------

FIR filtering can be much improved by using an explicit FFT based method. 

IIR filtering may be much improved by doing it in another way I suppose. 


HRTF-related stuff
------------------

Use the FIR filtering to implement HRTF related filtering.

Things that we don't need anymore
---------------------------------

FilterbankGroup, because we can use neurongroups and references to variable this is no longer needed. I think it is for the best.
FunctionFilterbank, the functions can always be implemented with the neurongroup.



