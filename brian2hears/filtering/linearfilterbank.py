from builtins import range, zip

import numpy as np

from brian2.codegen.cpp_prefs import get_compiler_and_args, update_for_cross_compilation
from brian2.utils.logger import std_silent, get_logger
from brian2.codegen.runtime.cython_rt.extension_manager import cython_extension_manager
from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject
try:
    import weave
except ImportError:
    try:
        from scipy import weave
    except ImportError:
        weave = None
try:
    import Cython
    if not CythonCodeObject.is_available():
        Cython = None
except ImportError:
    Cython = None
from scipy import signal, random
from .filterbank import Filterbank, RestructureFilterbank
from ..bufferable import Bufferable


__all__ = ['LinearFilterbank']

logger = get_logger('brian2.'+__name__) # bit of a hack, but fine

def _scipy_apply_linear_filterbank(b, a, x, zi):
    '''
    Parallel version of scipy lfilter command for a bank of n sequences of length 1
    
    In scipy.lfilter, you can apply a filter to multiple sounds at the same time,
    but you can't apply a bank of filters at the same time. This command does
    that. The coeffs b, a must be of shape (n,m,p), x must be of shape (s, n),
    and zi must be of shape (n,m-1,p). Here n is the number of channels in the
    filterbank, m is the order of the filter, p is the number of filters in
    a chain (cascade) to apply (you do first with (:,:,0) then (:,:,1), etc.),
    and s is the size of the buffer segment.
    '''
    alf_cache_b00 = [0]*zi.shape[2]
    alf_cache_a1 = [0]*zi.shape[2]
    alf_cache_b1 = [0]*zi.shape[2]
    alf_cache_zi00 = [0]*zi.shape[2]
    alf_cache_zi0 = [0]*zi.shape[2]
    alf_cache_zi1 = [0]*zi.shape[2]
    for curf in range(zi.shape[2]):
        alf_cache_b00[curf] = b[:, 0, curf]
        alf_cache_zi00[curf] = zi[:, 0, curf]
        alf_cache_b1[curf] = b[:, 1:b.shape[1], curf]
        alf_cache_a1[curf] = a[:, 1:b.shape[1], curf]
        alf_cache_zi0[curf] = zi[:, 0:b.shape[1]-1, curf]
        alf_cache_zi1[curf] = zi[:, 1:b.shape[1], curf]
    X = x.copy()
    output = np.empty_like(X)
    num_cascade = zi.shape[2]
    b_loop_size = b.shape[1]-2
    y = np.zeros(zi.shape[0])
    yr = np.reshape(y, (1, len(y))).T
    t = np.zeros(alf_cache_b1[0].shape, order='F')
    t2 = np.zeros(alf_cache_b1[0].shape, order='F')
    for sample, (x, o) in enumerate(zip(X, output)):
        xr = np.reshape(x, (1, len(x))).T
        for curf in range(num_cascade):
            #y = b[:, 0, curf]*x+zi[:, 0, curf]
            np.multiply(alf_cache_b00[curf], x, y)
            np.add(y, alf_cache_zi00[curf], y)
            #zi[:, :i-1, curf] = b[:, 1:i, curf]*xr+zi[:, 1:i, curf]-a[:, 1:i, curf]*yr
            np.multiply(alf_cache_b1[curf], xr, t)
            np.add(t, alf_cache_zi1[curf], t)
            np.multiply(alf_cache_a1[curf], yr, t2)
            np.subtract(t, t2, alf_cache_zi0[curf])
            u = x
            ur = xr
            x = y
            xr = yr
            y = u
            yr = ur
        #output[sample] = y
        o[:] = x
    return output


def _weave_apply_linear_filterbank(b, a, x, zi,
                                   cpp_compiler, extra_compile_args):
    if zi.shape[2]>1:
        # we need to do this so as not to alter the values in x in the C code below
        # but if zi.shape[2] is 1 there is only one filter in the chain and the
        # copy operation at the end of the C code will never happen.
        x = np.array(x, copy=True, order='C')
    else:
        # make sure that the array is in C-order
        x = np.asarray(x, order='C')
    y = np.empty_like(x)
    n, m, p = b.shape
    n1, m1, p1 = a.shape
    numsamples = x.shape[0]
    n = int(n)
    m = int(m)
    p = int(p)
    n1 = int(n1)
    m1 = int(m1)
    p1 = int(p1)
    numsamples = int(numsamples)
    if n1!=n or m1!=m or p1!=p or x.shape!=(numsamples, n) or zi.shape!=(n, m, p):
        raise ValueError('Data has wrong shape.')
    if numsamples>1 and not x.flags['C_CONTIGUOUS']:
        raise ValueError('Input data must be C_CONTIGUOUS')
    if not b.flags['F_CONTIGUOUS'] or not a.flags['F_CONTIGUOUS'] or not zi.flags['F_CONTIGUOUS']:
        raise ValueError('Filter parameters must be F_CONTIGUOUS')
    code = '''
    #define X(s,i) x[(s)*n+(i)]
    #define Y(s,i) y[(s)*n+(i)]
    #define A(i,j,k) a[(i)+(j)*n+(k)*n*m]
    #define B(i,j,k) b[(i)+(j)*n+(k)*n*m]
    #define Zi(i,j,k) zi[(i)+(j)*n+(k)*n*(m-1)]
    for(int s=0; s<numsamples; s++)
    {
        for(int k=0; k<p; k++)
        {
            double * rp_y = &(Y(s, 0));
            const double * rp_b1 = &(B(0, 0, k));
            double * rp_x = &(X(s, 0));
            double * rp_zi1 = &(Zi(0, 0, k));
            for(int j=0; j<n; j++)
                         rp_y[j] =   rp_b1[j]*rp_x[j] + rp_zi1[j];
            for(int i=0; i<m-2; i++)
            {
                const double * rp_b2 = &(B(0, i+1, k));
                const double * rp_a2 = &(A(0, i+1, k));
                double * rp_zi20 = &(Zi(0, i, k));
                double * rp_zi21 = &(Zi(0, i+1, k));
                for(int j=0;j<n;j++)
                    rp_zi20[j] = rp_b2[j]*rp_x[j] + rp_zi21[j] - rp_a2[j]*rp_y[j];
            }
            const double * rp_b3 = &(B(0, m-1, k));
            const double * rp_a3 = &(A(0, m-1, k));
            double * rp_zi3 = &(Zi(0, m-2, k));
            for(int j=0; j<n; j++)
                  rp_zi3[j] = rp_b3[j]*rp_x[j] - rp_a3[j]*rp_y[j];
            if(k<p-1)
                for(int j=0; j<n; j++)
                    rp_x[j] = rp_y[j];            
        }
    }
    '''
    weave.inline(code, ['b', 'a', 'x', 'zi', 'y', 'n', 'm', 'p', 'numsamples'],
                 compiler=cpp_compiler,
                 extra_compile_args=extra_compile_args)
    return y


class CythonLinearFilterbankApply(object):
    def __init__(self):
        self.compiler, self.extra_compile_args = get_compiler_and_args()
        code = '''
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True

import numpy as _numpy
cimport numpy as _numpy

cpdef parallel_lfilter(_numpy.ndarray[_numpy.float64_t, ndim=3] b,
                       _numpy.ndarray[_numpy.float64_t, ndim=3] a,
                       _numpy.ndarray[_numpy.float64_t, ndim=2] x,
                       _numpy.ndarray[_numpy.float64_t, ndim=3] zi,
                       _numpy.ndarray[_numpy.float64_t, ndim=2] y):
    cdef int n, m, p, n1, m1, p1, numsamples, s, k, i, j
    cdef double* py
    cdef double* px
    cdef double* pa
    cdef double* pb
    cdef double* pzi
    cdef double* pzi2
    n = b.shape[0]
    m = b.shape[1]
    p = b.shape[2]
    numsamples = x.shape[0]
    for s in range(numsamples):
        py = &(y[s, 0]) 
        px = &(x[s, 0])
        for k in range(p):
            pb = &(b[0, 0, k])
            pzi = &(zi[0, 0, k])
            for j in range(n):
                y[s, j] =   b[j, 0, k]*x[s, j] + zi[j, 0, k]
                # py[j] =   pb[j]*px[j] + pzi[j]
            for i in range(m-2):
                pa = &(a[0, i+1, k])
                pb = &(b[0, i+1, k])
                pzi = &(zi[0, i, k])                
                pzi2 = &(zi[0, i+1, k])
                for j in range(n):
                    # zi[j, i, k] = b[j, i+1, k]*x[s, j] + zi[j, i+1, k] - a[j, i+1, k]*y[s,j]
                    pzi[j] = pb[j]*px[j] + pzi2[j] - pa[j]*py[j]
            pa = &(a[0, m-1, k])
            pb = &(b[0, m-1, k])
            pzi = &(zi[0, m-2, k])                
            for j in range(n):
                # zi[j, m-2, k] = b[j, m-1, k]*x[s,j] - a[j, m-1, k]*y[s,j]
                pzi[j] = pb[j]*px[j] - pa[j]*py[j]
            if k<p-1:
                for j in range(n):
                    # x[s, j] = y[s, j]
                    px[j] = py[j]
        '''
        self.compiled_code = cython_extension_manager.create_extension(code,
                                                                       compiler=self.compiler,
                                                                       extra_compile_args=self.extra_compile_args)
    def __call__(self, b, a, x, zi):
        if zi.shape[2]>1:
            # we need to do this so as not to alter the values in x in the C code below
            # but if zi.shape[2] is 1 there is only one filter in the chain and the
            # copy operation at the end of the C code will never happen.
            x = np.array(x, copy=True, order='C')
        else:
            # make sure that the array is in C-order
            x = np.asarray(x, order='C')
        y = np.empty_like(x)
        n, m, p = b.shape
        n1, m1, p1 = a.shape
        numsamples = x.shape[0]
        n = int(n)
        m = int(m)
        p = int(p)
        n1 = int(n1)
        m1 = int(m1)
        p1 = int(p1)
        numsamples = int(numsamples)
        if n1 != n or m1 != m or p1 != p or x.shape != (numsamples, n) or zi.shape != (n, m, p):
            raise ValueError('Data has wrong shape.')
        if numsamples>1 and not x.flags['C_CONTIGUOUS']:
            raise ValueError('Input data must be C_CONTIGUOUS')
        if not b.flags['F_CONTIGUOUS'] or not a.flags['F_CONTIGUOUS'] or not zi.flags['F_CONTIGUOUS']:
            raise ValueError('Filter parameters must be F_CONTIGUOUS')
        self.compiled_code.parallel_lfilter(b, a, x, zi, y)
        return y


class LinearFilterbank(Filterbank):
    '''
    Generalised linear filterbank
    
    Initialisation arguments:

    ``source``
        The input to the filterbank, must have the same number of channels or
        just a single channel. In the latter case, the channels will be
        replicated.
    ``b``, ``a``
        The coeffs b, a must be of shape ``(nchannels, m)`` or
        ``(nchannels, m, p)``. Here ``m`` is
        the order of the filters, and ``p`` is the number of filters in a
        chain (first you apply ``[:, :, 0]``, then ``[:, :, 1]``, etc.).
    
    The filter parameters are stored in the modifiable attributes ``filt_b``,
    ``filt_a`` and ``filt_state`` (the variable ``z`` in the section below).
    
    Has one method:
    
    .. automethod:: decascade
    
    **Notes**
    
    These notes adapted from scipy's :func:`~scipy.signal.lfilter` function.
    
    The filterbank is implemented as a direct II transposed structure.
    This means that for a single channel and element of the filter cascade,
    the output y for an input x is defined by::

        a[0]*y[m] = b[0]*x[m] + b[1]*x[m-1] + ... + b[m]*x[0]
                              - a[1]*y[m-1] - ... - a[m]*y[0]

    using the following difference equations::

        y[i] = b[0]*x[i] + z[0,i-1]
        z[0,i] = b[1]*x[i] + z[1,i-1] - a[1]*y[i]
        ...
        z[m-3,i] = b[m-2]*x[i] + z[m-2,i-1] - a[m-2]*y[i]
        z[m-2,i] = b[m-1]*x[i] - a[m-1]*y[i]

    where i is the output sample number.

    The rational transfer function describing this filter in the
    z-transform domain is::
    
                                -1              -nb
                    b[0] + b[1]z  + ... + b[m] z
            Y(z) = --------------------------------- X(z)
                                -1              -na
                    a[0] + a[1]z  + ... + a[m] z
        
    '''
    def __init__(self, source, b, a):
        # Automatically duplicate mono input to fit the desired output shape
        if b.shape[0]!=source.nchannels:
            if source.nchannels!=1:
                raise ValueError('Can only automatically duplicate source channels for mono sources, use RestructureFilterbank.')
            source = RestructureFilterbank(source, b.shape[0])
        Filterbank.__init__(self, source)
        # Weave version of filtering requires Fortran ordering of filter params
        if len(b.shape)==2 and len(a.shape)==2:
            b = np.reshape(b, b.shape+(1,))
            a = np.reshape(a, a.shape+(1,))
        self.filt_b = np.array(b, order='F')
        self.filt_a = np.array(a, order='F')
        self.filt_state = np.zeros((b.shape[0], b.shape[1], b.shape[2]), order='F')
        self.use_weave = weave is not None
        if self.use_weave:
            logger.debug("Using weave for LinearFilterbank")
            self.cpp_compiler, self.extra_compile_args = get_compiler_and_args()
        else:
            self.use_cython = Cython is not None
            if self.use_cython:
                self.cython_func = CythonLinearFilterbankApply()

    def reset(self):
        self.buffer_init()
        
    def buffer_init(self):
        Filterbank.buffer_init(self)
        self.filt_state[:] = 0
    
    def buffer_apply(self, input):
        
        if self.use_weave:
            return _weave_apply_linear_filterbank(self.filt_b, self.filt_a, input,
                                                  self.filt_state, self.cpp_compiler,
                                                  self.extra_compile_args)
        elif self.use_cython:
            return self.cython_func(self.filt_b, self.filt_a, input, self.filt_state)
        else:
            return _scipy_apply_linear_filterbank(self.filt_b, self.filt_a, input,
                                                  self.filt_state)
        
    def decascade(self, ncascade=1):
        '''
        Reduces cascades of low order filters into smaller cascades of high order filters.
        
        ``ncascade`` is the number of cascaded filters to use, which should be
        a divisor of the original number.
        
        Note that higher order filters are often numerically unstable.
        '''
        n, m, p = self.filt_b.shape
        if p%ncascade!=0:
            raise ValueError('Number of cascades must be a divisor of original number of cascaded filters.')
        b = np.zeros((n, (m-1)*(p/ncascade)+1, ncascade))
        a = np.zeros((n, (m-1)*(p/ncascade)+1, ncascade))
        for i in range(n):
            for k in range(ncascade):
                bp = np.ones(1)
                ap = np.ones(1)
                for j in range(k*(p/ncascade), (k+1)*(p/ncascade)):
                    bp = np.polymul(bp, self.filt_b[i, ::-1, j])
                    ap = np.polymul(ap, self.filt_a[i, ::-1, j])
                bp = bp[::-1]
                ap = ap[::-1]
                a0 = ap[0]
                ap /= a0
                bp /= a0
                b[i, :len(bp), k] = bp
                a[i, :len(ap), k] = ap
        self.filt_b = np.array(b, order='F')
        self.filt_a = np.array(a, order='F')
        self.filt_state = np.zeros((b.shape[0], b.shape[1], b.shape[2]), order='F')
                
# # Use the GPU version if available
# try:
#     if get_global_preference('brianhears_usegpu'):
#         import pycuda
#         from gpulinearfilterbank import LinearFilterbank
#         use_gpu = True
#     else:
#         use_gpu = False
# except ImportError:
#     use_gpu = False
use_gpu = False
