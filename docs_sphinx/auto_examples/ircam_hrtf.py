#!/usr/bin/env python
'''
HRTFs
-----
Example showing the use of HRTFs in Brian hears. Note that you will need to
download the `.IRCAM_LISTEN` database and set the IRCAM_LISTEN environment variable to point to the location
where you saved it.
'''
from brian2 import *
from brian2hears import *
# Load database
hrtfdb = IRCAM_LISTEN()
hrtfset = hrtfdb.load_subject(hrtfdb.subjects[0])
# Select only the horizontal plane
hrtfset = hrtfset.subset(lambda elev: elev==0)
# Set up a filterbank
sound = whitenoise(10*ms)
fb = hrtfset.filterbank(sound)
# Extract the filtered response and plot
img = fb.process().T
img_left = img[:img.shape[0]//2, :]
img_right = img[img.shape[0]//2:, :]
subplot(121)
imshow(img_left, origin='lower left', aspect='auto',
       extent=(0, sound.duration/ms, 0, 360))
xlabel('Time (ms)')
ylabel('Azimuth')
title('Left ear')
subplot(122)
imshow(img_right, origin='lower left', aspect='auto',
       extent=(0, sound.duration/ms, 0, 360))
xlabel('Time (ms)')
ylabel('Azimuth')
title('Right ear')
show()
