from brian2 import *
from brian2hears import *

def test_basic_filterbanks():
    sound = sequence([whitenoise(100*ms),
                      tone(100*Hz, 100*ms)])
    res = FunctionFilterbank(sound, lambda x: abs(x)).process()
    res = Gammatone(sound, erbspace(50*Hz, 200*Hz, 4)).process()
    res = ApproximateGammatone(sound, erbspace(50 * Hz, 200 * Hz, 4), ones(4)*100*Hz).process()
    res = LogGammachirp(sound, erbspace(50 * Hz, 200 * Hz, 4)).process()
    res = LinearGammachirp(sound, erbspace(50 * Hz, 200 * Hz, 4), 10*ms).process()
    res = LinearGaborchirp(sound, erbspace(50 * Hz, 200 * Hz, 4), 10*ms).process()
    res = IIRFilterbank(sound, 1, passband=[200*Hz, 500*Hz], stopband=[100*Hz, 600*Hz],
                        gpass=10*dB, gstop=20*dB, btype='bandpass', ftype='ellip').process()
    res = Butterworth(sound, 1, 4, 100*Hz).process()
    res = LowPass(sound, 100*Hz).process()
    # plot(sound.times/ms, res)
    # show()

def test_filterbankgroup():
    sound1 = tone(1 * kHz, .1 * second)
    sound2 = whitenoise(.1 * second)
    sound = sound1 + sound2
    sound = sound.ramp()
    cf = erbspace(20 * Hz, 20 * kHz, 3000)
    cochlea = Gammatone(sound, cf)
    # Half-wave rectification and compression [x]^(1/3)
    ihc = FunctionFilterbank(cochlea, lambda x: 3 * clip(x, 0, Inf) ** (1.0 / 3.0))
    # Leaky integrate-and-fire model with noise and refractoriness
    eqs = '''
    dv/dt = (I-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1 (unless refractory)
    I : 1
    '''
    anf = FilterbankGroup(ihc, 'I', eqs, reset='v=0', threshold='v>1', refractory=5*ms)
    M = SpikeMonitor(anf)
    run(sound.duration)
    # plot(M.t/ms, M.i, ',k')
    # show()


def test_cochleagram():
    sound1 = tone(1 * kHz, .1 * second)
    sound2 = whitenoise(.1 * second)

    sound = sound1 + sound2
    sound = sound.ramp()

    cf = erbspace(20 * Hz, 20 * kHz, 3000)
    gammatone = Gammatone(sound, cf)
    cochlea = FunctionFilterbank(gammatone, lambda x: clip(x, 0, Inf) ** (1.0 / 3.0))
    lowpass = LowPass(cochlea, 10 * Hz)
    output = lowpass.process()

    # imshow(output.T, origin='lower left', aspect='auto', vmin=0)
    # show()


def test_firfilterbank():
    sound = whitenoise(100*ms)
    ir = Gammatone(click()[slice(0*ms, 100*ms)], [100, 200]*Hz).process()
    for linear in [True, False]:
        firfb = FIRFilterbank(sound, ir.T, use_linearfilterbank=linear)
        output1 = firfb.process()
        output2 = Gammatone(sound, [100, 200]*Hz).process()
        assert amax(abs(output1-output2))<1e-10
    # plot(sound.times/ms, output1)
    # plot(sound.times/ms, output2, '--')
    # show()


def test_fractionaldelayfilterbank():
    sound_orig = Sound(lambda t: sin(2*pi*30*Hz*t), duration=100*ms)
    delay = 0.5/sound_orig.samplerate
    fb = FractionalDelay(sound_orig, [delay])
    sound_delay = fb.process()
    sound_offset = Sound(lambda t: sin(2*pi*30*Hz*(t-(fb.delay_offset+delay)))*(t>=fb.delay_offset+delay), duration=100*ms)
    # plot(sound_orig.times/ms, sound_offset)
    # plot(sound_orig.times/ms, sound_delay)
    # show()


def test_dcgc():
    sound = whitenoise(100*ms)
    fb = DCGC(sound, [100, 200]*Hz)
    output = fb.process()
    # plot(sound.times/ms, output)
    # show()


def test_drnl():
    sound = whitenoise(100*ms)
    fb = DRNL(sound, [100, 200]*Hz)
    output = fb.process()
    # plot(sound.times/ms, output)
    # show()

if __name__=='__main__':
    # test_basic_filterbanks()
    # test_filterbankgroup()
    # test_cochleagram()
    # test_firfilterbank()
    # test_fractionaldelayfilterbank()
    # test_dcgc()
    test_drnl()
