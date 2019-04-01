import numpy as np

try:
    import weave
except ImportError:
    try:
        from scipy import weave
    except ImportError:
        weave = None
from scipy import signal, random
from math import factorial
from .filterbank import Filterbank,RestructureFilterbank
from .linearfilterbank import *
from .firfilterbank import *
from six.moves import range as xrange

from brian2 import second, DimensionMismatchError

__all__ = ['Cascade',
           'Gammatone',
           'ApproximateGammatone',
           'LogGammachirp',
           'LinearGammachirp',
           'LinearGaborchirp',
           'IIRFilterbank',
           'Butterworth',
           'AsymmetricCompensation',
           'LowPass',
           'asymmetric_compensation_coeffs',
           ]


    
class Gammatone(LinearFilterbank):
    '''
    Bank of gammatone filters.
    
    They are implemented as cascades of four 2nd-order IIR filters (this
    8th-order digital filter corresponds to a 4th-order gammatone filter).
    
    The approximated impulse response :math:`\\mathrm{IR}` is defined as follow
    :math:`\\mathrm{IR}(t)=t^3\\exp(-2\\pi b \\mathrm{ERB}(f)t)\\cos(2\\pi f t)`
    where :math:`\\mathrm{ERB}(f)=24.7+0.108 f` [Hz] is the equivalent
    rectangular bandwidth of the filter centered at :math:`f`.

    It comes from Slaney's exact gammatone implementation (Slaney, M., 1993,
    "An Efficient Implementation of the Patterson-Holdsworth 
    Auditory Filter Bank". Apple Computer Technical Report #35). The code is
    based on
    `Slaney's Matlab implementation <http://cobweb.ecn.purdue.edu/~malcolm/interval/1998-010/>`__.
    
    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
        
    ``cf``
        List or array of center frequencies.
        
    ``b=1.019``
        parameter which determines the bandwidth of the filters (and
        reciprocally the duration of its impulse response). In particular, the
        bandwidth = b.ERB(cf), where ERB(cf) is the equivalent bandwidth at
        frequency ``cf``. The default value of ``b`` to a best fit
        (Patterson et al., 1992). ``b`` can either be a scalar and will be the
        same for every channel or an array of the same length as ``cf``.
        
    ``erb_order=1``, ``ear_Q=9.26449``, ``min_bw=24.7``
        Parameters used to compute the ERB bandwidth.
        :math:`\\mathrm{ERB} = ((\mathrm{cf}/\mathrm{ear\\_Q})^{\\mathrm{erb}\\_\\mathrm{order}} + \\mathrm{min\\_bw}^{\\mathrm{erb}\\_\\mathrm{order}})^{(1/\\mathrm{erb}\\_\\mathrm{order})}`.
        Their default values are the ones recommended in
        Glasberg and Moore, 1990. 

    ``cascade=None``
        Specify 1 or 2 to use a cascade of 1 or 2 order 8 or 4 filters instead
        of 4 2nd order filters. Note that this is more efficient but may
        induce numerical stability issues.
    '''

    def __init__(self, source, cf, b=1.019, erb_order=1, ear_Q=9.26449,
                 min_bw=24.7, cascade=None):
        cf = np.atleast_1d(np.asarray(cf))
        self.cf = cf
        self.samplerate =  source.samplerate
        T = float(1/self.samplerate)
        self.b,self.erb_order,self.EarQ,self.min_bw=b,erb_order,ear_Q,min_bw
        erb = ((cf/ear_Q)**erb_order + min_bw**erb_order)**(1/erb_order)
        B = b*2*np.pi*erb
#        B = 2*np.pi*b

        A0 = T
        A2 = 0
        B0 = 1
        B1 = -2*np.cos(2*cf*np.pi*T)/np.exp(B*T)
        B2 = np.exp(-2*B*T)
        
        A11 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) + 2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T) / \

                np.exp(B*T))/2
        A12=-(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T)-2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/\
                np.exp(B*T))/2
        A13=-(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T)+2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/\
                np.exp(B*T))/2
        A14=-(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T)-2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/\
                np.exp(B*T))/2

        i=1j
        gain=abs((-2*np.exp(4*i*cf*np.pi*T)*T+\
                         2*np.exp(-(B*T)+2*i*cf*np.pi*T)*T*\
                                 (np.cos(2*cf*np.pi*T)-np.sqrt(3-2**(3./2))*\
                                  np.sin(2*cf*np.pi*T)))*\
                   (-2*np.exp(4*i*cf*np.pi*T)*T+\
                     2*np.exp(-(B*T)+2*i*cf*np.pi*T)*T*\
                      (np.cos(2*cf*np.pi*T)+np.sqrt(3-2**(3./2))*\
                       np.sin(2*cf*np.pi*T)))*\
                   (-2*np.exp(4*i*cf*np.pi*T)*T+\
                     2*np.exp(-(B*T)+2*i*cf*np.pi*T)*T*\
                      (np.cos(2*cf*np.pi*T)-\
                       np.sqrt(3+2**(3./2))*np.sin(2*cf*np.pi*T)))*\
                   (-2*np.exp(4*i*cf*np.pi*T)*T+2*np.exp(-(B*T)+2*i*cf*np.pi*T)*T*\
                   (np.cos(2*cf*np.pi*T)+np.sqrt(3+2**(3./2))*np.sin(2*cf*np.pi*T)))/\
                  (-2/np.exp(2*B*T)-2*np.exp(4*i*cf*np.pi*T)+\
                   2*(1+np.exp(4*i*cf*np.pi*T))/np.exp(B*T))**4)

        allfilts=np.ones(len(cf))

        self.A0, self.A11, self.A12, self.A13, self.A14, self.A2, self.B0, self.B1, self.B2, self.gain=\
            A0*allfilts, A11, A12, A13, A14, A2*allfilts, B0*allfilts, B1, B2, gain

        self.filt_a=np.dstack((np.array([np.ones(len(cf)), B1, B2]).T,)*4)
        self.filt_b=np.dstack((np.array([A0/gain, A11/gain, A2/gain]).T,
                         np.array([A0*np.ones(len(cf)), A12, np.zeros(len(cf))]).T,
                         np.array([A0*np.ones(len(cf)), A13, np.zeros(len(cf))]).T,
                         np.array([A0*np.ones(len(cf)), A14, np.zeros(len(cf))]).T))

        LinearFilterbank.__init__(self, source, self.filt_b, self.filt_a)
        if cascade is not None:
            self.decascade(cascade)

class ApproximateGammatone(LinearFilterbank):
    r'''
    Bank of approximate gammatone filters implemented as a cascade of ``order`` IIR gammatone filters.
    
    The filter is derived from the sampled version of the complex analog
    gammatone impulse response
    :math:`g_{\gamma}(t)=t^{\gamma-1} (\lambda e^{i \eta t})^{\gamma}` 
    where :math:`\gamma` corresponds to ``order``, :math:`\eta` defines the
    oscillation frequency ``cf``, and :math:`\lambda` defines the bandwidth
    parameter.
    
    The design is based on the Hohmann implementation as described in
    Hohmann, V., 2002, "Frequency analysis and synthesis using a Gammatone
    filterbank", Acta Acustica United with Acustica. The code is based on the
    Matlab gammatone implementation from
    `Meddis' toolbox <http://www.essex.ac.uk/psychology/psy/PEOPLE/meddis/webFolder08/WebIntro.htm>`__. 
    
    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
        
    ``cf``
        List or array of center frequencies.
        
    ``bandwidth``
        List or array of filters bandwidth corresponding, one for each cf.
        
    ``order=4``
        The number of 1st-order gammatone filters put in cascade, and therefore
        the order the resulting gammatone filters.
     '''
   
    def __init__(self, source, cf,  bandwidth,order=4):
        cf = np.asarray(np.atleast_1d(cf))
        bandwidth = np.asarray(np.atleast_1d(bandwidth))
        self.cf = cf
        self.samplerate =  source.samplerate
        dt = float(1/self.samplerate)
        phi = 2 * np.pi * bandwidth * dt
        theta = 2 * np.pi * cf * dt
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        alpha = -np.exp(-phi) * cos_theta
        b0 = np.ones(len(cf))
        b1 = 2 * alpha
        b2 = np.exp(-2 * phi)
        z1 = (1 + alpha * cos_theta) - (alpha * sin_theta) * 1j
        z2 = (1 + b1 * cos_theta) - (b1 * sin_theta) * 1j
        z3 = (b2 * np.cos(2 * theta)) - (b2 * np.sin(2 * theta)) * 1j
        tf = (z2 + z3) / z1
        a0 = abs(tf)
        a1 = alpha * a0   
        # we apply the same filters order times so we just duplicate them in the 3rd axis for the parallel_lfilter_step command
        self.filt_a = np.dstack((np.array([b0, b1, b2]).T,)*order)
        self.filt_b = np.dstack((np.array([a0, a1, np.zeros(len(cf))]).T,)*order)
        self.order = order
        
        LinearFilterbank.__init__(self,source, self.filt_b, self.filt_a)

           
class LogGammachirp(LinearFilterbank):
    r'''
    Bank of gammachirp filters with a logarithmic frequency sweep.
    
    The approximated impulse response :math:`\mathrm{IR}` is defined as follows:
    :math:`\mathrm{IR}(t)=t^3e^{-2\pi b \mathrm{ERB}(f)t}\cos(2\pi (f t +c\cdot\ln(t))`
    where :math:`\mathrm{ERB}(f)=24.7+0.108 f` [Hz] is the equivalent
    rectangular bandwidth of the filter centered at :math:`f`.
    
    The implementation is a cascade of 4 2nd-order IIR gammatone filters 
    followed by a cascade of ncascades 2nd-order asymmetric compensation filters
    as introduced in Unoki et al. 2001, "Improvement of an IIR asymmetric 
    compensation gammachirp filter". 

    Initialisation parameters:
    
    ``source``
        Source sound or filterbank.
        
    ``f``
        List or array of the sweep ending frequencies
        (:math:`f_{\mathrm{instantaneous}}=f+c/t`). 
        
    ``b=1.019``
        Parameters which determine the duration of the impulse response.
        ``b`` can either be a scalar and will be the same for every channel or
        an array with the same length as ``f``.
        
    ``c=1``
        The glide slope (or sweep rate) given in Hz/second. The trajectory of
        the instantaneous frequency towards f is an upchirp when c<0 and a 
        downchirp when c>0.
        ``c`` can either be a scalar and will be the same for every channel or
        an array with the same length as ``f``.
        
    ``ncascades=4``
        Number of times the asymmetric compensation filter is put in cascade.
        The default value comes from Unoki et al. 2001. 
    '''
      
    def __init__(self, source, f,b=1.019,c=1,ncascades=4):
        f = np.atleast_1d(np.asarray(f))
        self.f = f
        self.samplerate= source.samplerate
        
        self.c=c
        self.b=b
        gammatone = Gammatone(source, f, b)

        self.gammatone_filt_b=gammatone.filt_b
        self.gammatone_filt_a=gammatone.filt_a

        ERBw=24.7*(4.37e-3*f+1.)

        p0=2
        p1=1.7818*(1-0.0791*b)*(1-0.1655*abs(c))
        p2=0.5689*(1-0.1620*b)*(1-0.0857*abs(c))
        p3=0.2523*(1-0.0244*b)*(1+0.0574*abs(c))
        p4=1.0724

        self.asymmetric_filt_b=np.zeros((len(f),3, ncascades))
        self.asymmetric_filt_a=np.zeros((len(f),3, ncascades))

        self.asymmetric_filt_b, self.asymmetric_filt_a = asymmetric_compensation_coeffs(self.samplerate,
                                                                                        f,
                                                                                        self.asymmetric_filt_b,
                                                                                        self.asymmetric_filt_a,
                                                                                        b, c, p0, p1, p2, p3, p4)

        #concatenate the gammatone filter coefficients so that everything is in cascade in each frequency channel
        self.filt_b = np.concatenate([self.gammatone_filt_b,
                                      self.asymmetric_filt_b], axis=2)
        self.filt_a = np.concatenate([self.gammatone_filt_a,
                                      self.asymmetric_filt_a], axis=2)
        
        LinearFilterbank.__init__(self, source, self.filt_b,self.filt_a)


class LinearGammachirp(FIRFilterbank):
    r'''
    Bank of gammachirp filters with linear frequency sweeps and gamma envelope
    as described in Wagner et al. 2009, "Auditory responses in the barn owl's
    nucleus laminaris to clicks: impulse response and signal analysis of
    neurophonic potential", J. Neurophysiol.
    
    The impulse response :math:`\mathrm{IR}` is defined as follow
    :math:`\mathrm{IR}(t)=t^3e^{-t/\sigma}\cos(2\pi (f t +c/2 t^2)+\phi)`
    where :math:`\sigma` corresponds to ``time_constant`` and :math:`\phi` to
    ``phase`` (see definition of parameters).

    Those filters are implemented as FIR filters using truncated time
    representations of gammachirp functions as the impulse response. The impulse
    responses, which need to have the same length for every channel, have a
    duration of 15 times the biggest time constant. The length of the impulse
    response is therefore ``15*max(time_constant)*sampling_rate``.  The impulse
    responses are normalized with respect to the transmitted power, i.e.
    the rms of the filter taps is 1.
    
    Initialisation parameters:
    
    ``source``
        Source sound or filterbank.
        
    ``f``
        List or array of the sweep starting frequencies
        (:math:`f_{\mathrm{instantaneous}}=f+ct`).
        
    ``time_constant``
        Determines the duration of the envelope and consequently the length of
        the impulse response.
        
    ``c=1``
        The glide slope (or sweep rate) given in Hz/second. The time-dependent
        instantaneous frequency is ``f+c*t`` and is therefore going upward when
        c>0 and downward when c<0. ``c`` can either be a scalar and will be the
        same for every channel or an array with the same length as ``f``.
        
    ``phase=0``
        Phase shift of the carrier.
        
    Has attributes:
    
    ``length_impulse_response`` 
        Number of samples in the impulse responses.
        
    ``impulse_response``
        Array of shape ``(nchannels, length_impulse_response)`` with each row
        being an impulse response for the corresponding channel.
    '''
    def __init__(self, source, f, time_constant, c=1, phase=0):
        
        self.f=f=np.asarray(np.atleast_1d(f))
        self.c=c=np.atleast_1d(c)
        self.phase=phase=np.atleast_1d(phase)
        self.time_constant=time_constant=np.asarray(np.atleast_1d(time_constant))
        if len(time_constant)==1:
            time_constant=time_constant*np.ones(len(f))
        if len(c)==1:
            c=c*np.ones(len(f))
        if len(phase)==1:
            phase=phase*np.ones(len(f))
        self.samplerate= source.samplerate

        Tcst_max=max(time_constant)

        t_start=float(-Tcst_max*3*second)
        t=np.arange(t_start,-4*t_start,float(1./self.samplerate))

        self.impulse_response=np.zeros((len(f),len(t)))
                                    
        for ich in xrange(len(f)):
            env=(t-t_start)**3*np.exp(-(t-t_start)/time_constant[ich])        
            self.impulse_response[ich,:]=env*np.cos(2*np.pi*(f[ich]*t+c[ich]/2*t**2)+phase[ich])
#            self.impulse_response[ich,:]=self.impulse_response[ich,:]/np.sqrt(sum(self.impulse_response[ich,:]**2))
            self.impulse_response[ich,:]=self.impulse_response[ich,:]/np.sum(abs(self.impulse_response[ich,:]))        


        FIRFilterbank.__init__(self,source, self.impulse_response)

class LinearGaborchirp(FIRFilterbank):
    r'''
    Bank of gammachirp filters with linear frequency sweeps and gaussian envelope
    as described in Wagner et al. 2009, "Auditory responses in the barn owl's 
    nucleus laminaris to clicks: impulse response and signal analysis of
    neurophonic potential", J. Neurophysiol.
    
    The impulse response :math:`\mathrm{IR}` is defined as follows:
    :math:`\mathrm{IR}(t)=e^{-t/2\sigma^2}\cos(2\pi (f t +c/2 t^2)+\phi)`,
    where :math:`\sigma` corresponds to ``time_constant`` and :math:`\phi` to
    ``phase`` (see definition of parameters).
    
    These filters are implemented as FIR filters using truncated time
    representations of gammachirp functions as the impulse response. The impulse
    responses, which need to have the same length for every channel, have a
    duration of 12 times the biggest time constant. The length of the impulse
    response is therefore ``12*max(time_constant)*sampling_rate``. The envelope
    is a gaussian function (Gabor filter).  The impulse responses are normalized
    with respect to the transmitted  power, i.e. the rms of the filter taps is
    1.
    
    Initialisation parameters:
    
    ``source``
        Source sound or filterbank.
        
    ``f``
        List or array of the sweep starting frequencies
        (:math:`f_{\mathrm{instantaneous}}=f+c*t`).
        
    ``time_constant``
        Determines the duration of the envelope and consequently the length of
        the impluse response.
        
    ``c=1``
        The glide slope (or sweep rate) given ins Hz/second. The time-dependent
        instantaneous frequency is ``f+c*t`` and is therefore going upward when
        c>0 and downward when c<0. ``c`` can either be a scalar and will be the
        same for every channel or an array with the same length as ``f``.
        
    ``phase=0``
        Phase shift of the carrier.
        
    Has attributes:
    
    ``length_impulse_response`` 
        Number of sample in the impulse responses.
        
    ``impulse_response``
        Array of shape ``(nchannels, length_impulse_response)`` with each row
        being an impulse response for the corresponding channel.
    '''
    def __init__(self,source, f, time_constant, c=1, phase=0):
        self.f=f=np.asarray(np.atleast_1d(f))
        self.c=c=np.atleast_1d(c)
        self.phase=phase=np.atleast_1d(phase)
        self.time_constant=time_constant=np.asarray(np.atleast_1d(time_constant))
        if len(time_constant)==1:
            time_constant=np.asarray(time_constant*np.ones(len(f)))
        if len(c)==1:
            c=c*np.ones(len(f))
        if len(phase)==1:
            phase=phase*np.ones(len(f))
        self.samplerate = source.samplerate
        
        Tcst_max=max(time_constant)

        t_start=float(-Tcst_max*6*second)
        t=np.arange(t_start,-t_start,float(1./self.samplerate))

        self.impulse_response=np.zeros((len(f),len(t)))
                                    
        for ich in xrange(len(f)):
            env=np.exp(-(t/(2*time_constant[ich]))**2)   
            self.impulse_response[ich,:]=env*np.cos(2*np.pi*(f[ich]*t+c[ich]/2*t**2)+phase[ich])
            self.impulse_response[ich,:]=self.impulse_response[ich,:]/np.sqrt(np.sum(self.impulse_response[ich,:]**2))

        FIRFilterbank.__init__(self, source, self.impulse_response)

class IIRFilterbank(LinearFilterbank):
    '''
    Filterbank of IIR filters. The filters can be low, high, bandstop or
    bandpass and be of type Elliptic, Butterworth, Chebyshev etc. The
    ``passband``  and ``stopband`` can be scalars (for low or high pass) or
    pairs of parameters (for stopband and passband) yielding similar filters for
    every channel. They can also be arrays of shape ``(1, nchannels)`` for low
    and high pass or ``(2, nchannels)`` for stopband and passband yielding
    different filters along channels. This class uses the scipy iirdesign
    function to generate filter coefficients for every channel. 
    
    See the documentation for scipy.signal.iirdesign for more details.
    
    Initialisation parameters:
    
    ``samplerate``
        The sample rate in Hz.
        
    ``nchannels``
        The number of channels in the bank
        
    ``passband``, ``stopband``
        The edges of the pass and stop bands in Hz. For lowpass and highpass
        filters, in the case of similar filters for each channel, they are
        scalars and passband<stopband for low pass or stopband>passband for a
        highpass. For a bandpass or bandstop filter, in the case of similar
        filters for each channel, make passband and stopband a list with two
        elements, e.g. for a bandpass have ``passband=[200*Hz, 500*Hz]`` and
        ``stopband=[100*Hz, 600*Hz]``. ``passband`` and ``stopband`` can also be
        arrays of shape ``(1, nchannels)`` for low and high pass or
        ``(2, nchannels)`` for stopband and passband yielding different filters
        along channels.
        
    ``gpass``
        The maximum loss in the passband in dB. Can be a scalar or an array of
        length ``nchannels``.
        
    ``gstop``
        The minimum attenuation in the stopband in dB. Can be a scalar or an
        array of length ``nchannels``.
        
    ``btype``
        One of 'low', 'high', 'bandpass' or 'bandstop'.
    
    ``ftype``
        The type of IIR filter to design:
        'ellip' (elliptic),
        'butter' (Butterworth),
        'cheby1' (Chebyshev I),
        'cheby2' (Chebyshev II),
        'bessel' (Bessel).
    
    '''
    
    def __init__(self, source, nchannels, passband, stopband, gpass, gstop, btype, ftype):

        Wpassband = np.asarray(np.atleast_1d(passband).copy())
        Wstopband = np.asarray(np.atleast_1d(stopband).copy())
        gpass = np.atleast_1d(gpass)
        gstop = np.atleast_1d(gstop)
        
        self.samplerate=source.samplerate
        if Wpassband.shape != Wstopband.shape:
            raise Exception('passband and stopband must contain the same number of ent')
        try:
            Wpassband=Wpassband/float(self.samplerate)*2+0.0    # wn=1 corresponding to half the sample rate
            Wstopband=Wstopband/float(self.samplerate)*2+0.0
        except DimensionMismatchError:
            raise DimensionMismatchError('IIRFilterbank passband, stopband parameters must be in Hz')
        
        # now design filterbank      
        if btype=='low' or btype=='high':
            if len(Wpassband)==1:     #if there is only one Wn value for all channel just repeat it
                self.filt_b, self.filt_a = signal.iirdesign(Wpassband, Wstopband, gpass, gstop, ftype=ftype)
                self.filt_b=np.kron(np.ones((nchannels,1)),self.filt_b)
                self.filt_a=np.kron(np.ones((nchannels,1)),self.filt_a)
            else:               #else make nchannels different filters
                if len(gstop) != nchannels: #if the ripple parameters are scalar make them as long as the number of channels
                    gpass=np.repeat(gpass,nchannels)
                if len(gstop) != nchannels:
                    gstop=np.repeat(gstop,nchannels)
                order=0
                filt_b, filt_a =[1]*nchannels,[1]*nchannels  
                for i in xrange((nchannels)): #generate the different filter coeffcients
                    filt_b[i], filt_a[i] = signal.iirdesign(Wpassband[i], Wstopband[i], gpass[i], gstop[i], ftype=ftype)
                if len(filt_b[i])>order: #take the highst order of them to be the size of the filter coefficient matrix
                    order=len(filt_b[i])
                self.filt_b=np.zeros((nchannels,order))
                self.filt_a=np.zeros((nchannels,order))
                for i in xrange((nchannels)): #fill the coefficient matrix 
                    self.filt_b[i,:len(filt_b[i])], self.filt_a[i,:len(filt_a[i])] = filt_b[i],filt_a[i]
        else:
            if Wpassband.ndim==1:     #if there is only one Wn pair of values for all channel just repeat it
                self.filt_b, self.filt_a = signal.iirdesign(Wpassband, Wstopband, gpass, gstop, ftype=ftype)
                self.filt_b=np.kron(np.ones((nchannels,1)),self.filt_b)
                self.filt_a=np.kron(np.ones((nchannels,1)),self.filt_a)
            else:   
                if len(gstop) != nchannels:#if the ripple parameters are scalar make them as long as the number of channels
                    gpass=np.repeat(gpass,nchannels)
                if len(gstop) != nchannels:
                    gstop=np.repeat(gstop,nchannels)
                order=0
                filt_b, filt_a =[1]*nchannels,[1]*nchannels
                for i in xrange((nchannels)):#take the highst order of them to be the size of the filter coefficient matrix
                    filt_b[i], filt_a[i] = signal.iirdesign(Wpassband[:,i], Wstopband[:,i], gpass[i], gstop[i], ftype=ftype)
                if len(filt_b[i])>order:
                    order=len(filt_b[i])
                self.filt_b=np.zeros((nchannels,order))
                self.filt_a=np.zeros((nchannels,order))
                for i in xrange((nchannels)):#fill the coefficient matrix 
                    self.filt_b[i,:len(filt_b[i])], self.filt_a[i,:len(filt_a[i])] = filt_b[i],filt_a[i]
                    
        
        self.filt_a=self.filt_a.reshape(self.filt_a.shape[0],self.filt_a.shape[1],1)
        self.filt_b=self.filt_b.reshape(self.filt_b.shape[0],self.filt_b.shape[1],1)  
        self.nchannels = nchannels
        self.passband = passband
        self.stopband = stopband
        self.gpass = gpass
        self.gstop = gstop
        self.ftype= ftype
        self.order= self.filt_a.shape[1]-1
        LinearFilterbank.__init__(self,source, self.filt_b, self.filt_a) 


class Butterworth(LinearFilterbank):
    '''
    Filterbank of  low, high, bandstop or bandpass  Butterworth filters. 
    The cut-off frequencies or the band frequencies can either be the same for
    each channel or different along channels.
    
    Initialisation parameters:
    
    ``samplerate``
        Sample rate.
        
    ``nchannels``
        Number of filters in the bank.
        
    ``order``
        Order of the filters.
        
    ``fc``
        Cutoff parameter(s) in Hz. For the case of a lowpass or highpass
        filterbank, ``fc`` is either a scalar (thus the same value for all of
        the channels) or an array  of length ``nchannels``. For the case of a
        bandpass or bandstop, ``fc`` is either a pair of scalar defining the
        bandpass or bandstop (thus the same values for all of the channels) or
        an array of shape ``(2, nchannels)`` to define a pair for every channel.
        
    ``btype``
        One of 'low', 'high', 'bandpass' or 'bandstop'.
    '''

    def __init__(self,source, nchannels, order, fc, btype='low'):
        Wn = np.asarray(np.atleast_1d(fc)).copy() #Scalar inputs are converted to 1-dimensional arrays
        self.samplerate = source.samplerate
        Wn= Wn/float(self.samplerate)*2    # wn=1 corresponding to half the sample rate

        if btype=='low' or btype=='high':
            self.filt_b=np.zeros((nchannels,order+1))
            self.filt_a=np.zeros((nchannels,order+1))
            if len(Wn)==1:     #if there is only one Wn value for all channel just repeat it
                self.filt_b, self.filt_a = signal.butter(order, Wn, btype=btype)
                self.filt_b=np.kron(np.ones((nchannels,1)),self.filt_b)
                self.filt_a=np.kron(np.ones((nchannels,1)),self.filt_a)
            else:               #else make nchannels different filters
                for i in xrange((nchannels)):
                    self.filt_b[i,:], self.filt_a[i,:] = signal.butter(order, Wn[i], btype=btype)
        else:
            self.filt_b=np.zeros((nchannels,2*order+1))
            self.filt_a=np.zeros((nchannels,2*order+1))
            if Wn.ndim==1:     #if there is only one Wn pair of values for all channel just repeat it
                self.filt_b, self.filt_a = signal.butter(order, Wn, btype=btype)
                self.filt_b=np.kron(np.ones((nchannels,1)),self.filt_b)
                self.filt_a=np.kron(np.ones((nchannels,1)),self.filt_a)
            else:   
                for i in xrange((nchannels)):
                    self.filt_b[i,:], self.filt_a[i,:] = signal.butter(order, Wn[:,i], btype=btype)   
                    
                    
        self.filt_a=self.filt_a.reshape(self.filt_a.shape[0],self.filt_a.shape[1],1)
        self.filt_b=self.filt_b.reshape(self.filt_b.shape[0],self.filt_b.shape[1],1)  
        self.nchannels = nchannels    
        LinearFilterbank.__init__(self,source, self.filt_b, self.filt_a) 


class LowPass(LinearFilterbank):
    '''
    Bank of 1st-order lowpass filters
    
    The code is based on the code found in the
    `Meddis toolbox <http://www.essex.ac.uk/psychology/psy/PEOPLE/meddis/webFolder08/WebIntro.htm>`__. 
    It was implemented here to be used in the DRNL cochlear model implementation.

    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
        
    ``fc``
        Value, list or array (with length = number of channels) of cutoff
        frequencies.
    '''
    def __init__(self,source,fc):
        fc = np.asarray(np.atleast_1d(fc))
        if len(fc)==1:
            fc = fc*np.ones(source.nchannels)
        nchannels=len(fc)
        self.samplerate= source.samplerate
        dt=float(1./self.samplerate)

        self.filt_b=np.zeros((nchannels, 2, 1))
        self.filt_a=np.zeros((nchannels, 2, 1))
        tau=1/(2*np.pi*fc)
        self.filt_b[:,0,0]=dt/tau
        self.filt_b[:,1,0]=0*np.ones(nchannels)
        self.filt_a[:,0,0]=1*np.ones(nchannels)
        self.filt_a[:,1,0]=-(1-dt/tau)
        LinearFilterbank.__init__(self,source, self.filt_b, self.filt_a) 

      
class Cascade(LinearFilterbank):
    '''
    Cascade of ``n`` times a linear filterbank. 
    
    Initialised with arguments:
    
    ``source``
        Source of the new filterbank.
        
    ``filterbank``
        Filterbank object to be put in cascade
        
    ``n``
        Number of cascades
    '''
    
    def __init__(self,source, filterbank,n):
        b=filterbank.filt_b
        a=filterbank.filt_a
        self.samplerate =  source.samplerate
        self.nchannels=filterbank.nchannels
        self.filt_b=np.zeros((b.shape[0], b.shape[1],n))
        self.filt_a=np.zeros((a.shape[0], a.shape[1],n))
        for i in range((n)):
            self.filt_b[:,:,i]=b[:,:,0]
            self.filt_a[:,:,i]=a[:,:,0]
            
        LinearFilterbank.__init__(self, source,self.filt_b, self.filt_a)  
        
class AsymmetricCompensation(LinearFilterbank):
    '''
    Bank of asymmetric compensation filters.
    
    Those filters are meant to be used in cascade with gammatone filters to
    approximate gammachirp filters (Unoki et al., 2001, Improvement of
    an IIR asymmetric compensation gammachirp filter, Acoust. Sci. & Tech.).
    They are implemented a a cascade of low order filters. The code 
    is based on the implementation found in the
    `AIM-MAT toolbox <http://www.pdn.cam.ac.uk/groups/cnbh/aimmanual/index.html>`__.

    Initialised with arguments:
    
    ``source``
        Source of the filterbank.
        
    ``f``
        List or array of the cut off frequencies.
        
    ``b=1.019``
        Determines the duration of the impulse response.
        Can either be a scalar and will be the same for every channel or
        an array with the same length as ``cf``.
        
    ``c=1``
        The glide slope when this filter is used to implement a gammachirp.
        Can either be a scalar and will be the same for every channel or
        an array with the same length as ``cf``.
        
    ``ncascades=4``
        The number of time the basic filter is put in cascade.
     '''
     
    def __init__(self, source, f,b=1.019, c=1,ncascades=4):
        
        f = np.asarray(np.atleast_1d(f))
        self.f = f
        self.samplerate =  source.samplerate     
        ERBw=24.7*(4.37e-3*f+1.)
        p0=2
        p1=1.7818*(1-0.0791*b)*(1-0.1655*abs(c))
        p2=0.5689*(1-0.1620*b)*(1-0.0857*abs(c))
        p3=0.2523*(1-0.0244*b)*(1+0.0574*abs(c))
        p4=1.0724

        self.filt_b=np.zeros((len(f), 3, ncascades))
        self.filt_a=np.zeros((len(f), 3, ncascades))

        for k in np.arange(ncascades):

            r=np.exp(-p1*(p0/p4)**(k)*2*np.pi*b*ERBw/float(self.samplerate)) #k instead of k-1 because range 0 N-1
            Df=(p0*p4)**(k)*p2*c*b*ERBw

            phi=2*np.pi*np.maximum((f+Df), 0)/float(self.samplerate)
            psy=2*np.pi*np.maximum((f-Df), 0)/float(self.samplerate)

            ap=np.vstack((np.ones(r.shape),-2*r*np.cos(phi), r**2)).T
            bz=np.vstack((np.ones(r.shape),-2*r*np.cos(psy), r**2)).T

            fn=f#+ compensation_filter_order* p3 *c *b *ERBw/4;

            vwr=np.exp(1j*2*np.pi*fn/float(self.samplerate))
            vwrs=np.vstack((np.ones(vwr.shape), vwr, vwr**2)).T

            ##normilization stuff
            nrm=np.abs(np.sum(vwrs*ap, 1)/np.sum(vwrs*bz, 1))
            
            bz=bz*np.tile(nrm,[3,1]).T
            self.filt_b[:, :, k]=bz
            self.filt_a[:, :, k]=ap

        LinearFilterbank.__init__(self, source, self.filt_b, self.filt_a)



def asymmetric_compensation_coeffs(samplerate,fr,filt_b,filt_a,b,c,p0,p1,p2,p3,p4):
    '''
    This function is used to generated the coefficient of the asymmetric
    compensation filter used for the gammachirp implementation.
    '''
    samplerate = float(samplerate)
    ERBw=24.7*(4.37e-3*fr+1.)
    nbr_cascade=4
    for k in np.arange(nbr_cascade):
        r=np.exp(-p1*(p0/p4)**(k)*2*np.pi*b*ERBw/samplerate) #k instead of k-1 because range 0 N-1

        Dfr=(p0*p4)**(k)*p2*c*b*ERBw

        phi=2*np.pi*np.maximum((fr+Dfr), 0)/samplerate
        psy=2*np.pi*np.maximum((fr-Dfr), 0)/samplerate

        ap=np.vstack((np.ones(r.shape),-2*r*np.cos(phi), r**2)).T
        bz=np.vstack((np.ones(r.shape),-2*r*np.cos(psy), r**2)).T

        vwr=np.exp(1j*2*np.pi*fr/samplerate)
        vwrs=np.vstack((np.ones(vwr.shape), vwr, vwr**2)).T

        ##normilization stuff
        nrm=np.abs(np.sum(vwrs*ap, 1)/np.sum(vwrs*bz, 1))
        bz=bz*np.tile(nrm,[3,1]).T
        filt_b[:, :, k]=bz
        filt_a[:, :, k]=ap

    return filt_b,filt_a
