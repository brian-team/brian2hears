from brian2 import ms

from brian2hears import *
import os

def test_ircam_listen():
    listen_dir = os.getenv('IRCAM_LISTEN')
    if listen_dir is None:
        return
    hrtfdb = IRCAM_LISTEN(listen_dir)
    subject = hrtfdb.subjects[0]
    hrtfset = hrtfdb.load_subject(subject)
    # Select only the horizontal plane
    hrtfset = hrtfset.subset(lambda elev: elev==0)
    # Set up a filterbank
    sound = whitenoise(10*ms)
    fb = hrtfset.filterbank(sound)
    # Extract the filtered response and plot
    img = fb.process().T
    # img_left = img[:img.shape[0]/2, :]
    # img_right = img[img.shape[0]/2:, :]
    # subplot(121)
    # imshow(img_left, origin='lower', aspect='auto',
    #        extent=(0, sound.duration/ms, 0, 360))
    # xlabel('Time (ms)')
    # ylabel('Azimuth')
    # title('Left ear')
    # subplot(122)
    # imshow(img_right, origin='lower', aspect='auto',
    #        extent=(0, sound.duration/ms, 0, 360))
    # xlabel('Time (ms)')
    # ylabel('Azimuth')
    # title('Right ear')
    # show()

def test_headless_database():
    hrtfdb = HeadlessDatabase(24)
    subject = hrtfdb.subjects[0]
    hrtfset = hrtfdb.load_subject(subject)
    # Set up a filterbank
    sound = whitenoise(10*ms)
    fb = hrtfset.filterbank(sound)
    # Extract the filtered response and plot
    img = fb.process().T
    # img_left = img[:img.shape[0]/2, :]
    # img_right = img[img.shape[0]/2:, :]
    # subplot(121)
    # imshow(img_left, origin='lower', aspect='auto',
    #        extent=(0, sound.duration/ms, 0, 360))
    # xlabel('Time (ms)')
    # ylabel('Azimuth')
    # title('Left ear')
    # subplot(122)
    # imshow(img_right, origin='lower', aspect='auto',
    #        extent=(0, sound.duration/ms, 0, 360))
    # xlabel('Time (ms)')
    # ylabel('Azimuth')
    # title('Right ear')
    # show()


if __name__=='__main__':
    test_ircam_listen()
    test_headless_database()
