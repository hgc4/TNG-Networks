import warnings
warnings.filterwarnings("ignore")
import h5py
import numpy as np
import pandas as pd
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import os
import sys
from time import monotonic
from matplotlib.legend_handler import HandlerPathCollection as leghand
from scipy.stats import spearmanr, chisquare, iqr
from collections import Counter
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import median_filter as mf
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
mpl.rcParams.update({'errorbar.capsize': 4})
mpl.rcParams['image.cmap'] = 'nipy_spectral'
mpl.rcParams['image.origin'] = 'lower'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['hist.bins'] = 100
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
forbidden_names = ['fluxlines']

TNGtime, TNGredshift, TNGscalefactor, TNGHubble = np.load('TNGTime.npy') # time domain of the TNG simulations in four formats
age = max(TNGtime) - TNGtime # converting cosmic time to lookback time

def distance(A, B, bounds=None, shift=None):
    '''
    Computes the Pythagorean distance between coordinates, optionally with specified periodic boundary conditions.
    '''
    C = np.abs(A - B)
    if shift:
        C -= shift
    if bounds:
        C = np.where(C > 0.5 * bounds, C - bounds, C)
    return np.sqrt((C ** 2).sum(axis=-1))

def n_weighted_moment(values, weights=None, n=1):
    '''
    Calculates the nth weighted moment (mean, variance, etc.) from a set of datapoints and optional weights.
    '''
    if weights is None:
        weights = np.ones(np.shape(values))
    
    assert n > 0, "n must be positive"
    assert values.shape == weights.shape, "values and weights must be equal in shape"
    
    w_avg = np.average(values, weights = weights)
    w_var = np.sum(weights * (values - w_avg)**2)/np.sum(weights)
    w_std = np.sqrt(w_var)

    if n==1:
        return w_avg
    elif n==2:
        return w_var
    else:
        return np.sum(weights * ((values - w_avg)/w_std)**n)/np.sum(weights)

def readfile(filepath):
    '''
    Code for importing a numerical text file to a 1D or 2D numpy array. Useful for very large text files.
    '''
    filedata=[]
    with open(filepath) as infile:
        for line in infile:
            fileline = np.asarray([float(entry) for entry in line.split()])
            filedata.append(fileline)
    return np.asarray(filedata)

def nzmean(dat, axis=1, threshold=1e-3):
    '''
    Computes the mean of datapoints while ignoring all values below and including a specified threshold.
    '''
    Dat = dat.copy()
    Dat[Dat<=threshold] = np.nan
    return np.nanmean(Dat, axis=axis)

def lcm(int1, int2):
    '''
    Computes the lowest common multiple of the numbers int1 and int2.
    '''
    greater = max([int1, int2])
    while(True):
        if((greater % int1 == 0) and (greater % int2 == 0)):
            lcm = greater
            break
        greater += 1
    return lcm

def IntWithCommas(x):
    '''
    Returns an integer as a string with thousand-separating commas.
    '''
    if x < 0:
        return '-' + IntWithCommas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "%d%s" % (x, result)

def rmNanInf(arr):
    '''
    Replaces all NaN or infinite array elements with zeros.
    '''
    arr = np.where(np.isnan(arr), 0, arr)
    arr = np.where(np.isinf(arr), 0, arr)
    return arr

def verb(index, length, numperc=100, string=None):
    '''
    Verbose output for use in loops. Prints percentage of completion of said loop.
    '''
    A = int(index * numperc / length)
    B = int((1 + index) * numperc / length)
    C = 100. / float(numperc)
    if C == int(C):
        C = int(C)
    if B > A:
        if string:
            print('{}: {}% Complete.'.format(string, C*B))
        else:
            print('{}% Complete.'.format(C*B))

def virv(mass, halfrad):
    '''
    Proxy for the virial velocity of a dark matter subhalo, in terms of its total mass and half-mass radius.
    '''
    return np.nan_to_num(np.sqrt(mass / halfrad), posinf=0, neginf=0)

def growpast(x, xc, flip=True):
    '''
    Returns the index of a 1D array corresponding to the first element to exceed a specified threshold.
    '''
    Range = np.arange(len(x))
    if flip:
        Range = Range[::-1]
    result = 0
    for o in Range:
        if x[o] > xc:
            result += o
            break
    return result

def s2ms(t, String=True, Round=True):
    '''
    Converts a time given in seconds to minutes and seconds, or to hours, minutes and seconds if longer than or equal to 1 hour. Returns either a numpy array or a string.
    '''
    if Round:
        t = int(round(t))
    if t < 3600:
        m = int(float(t)/60.)
        s = t - (m * 60)
        if String:
            return '{}m, {}s'.format(m, s)
        else:
            if Round:
                return np.asarray((m, s), dtype=int)
            else:
                return np.asarray((m, s), dtype=float)
    else:
        h = int(float(t)/3600.)
        th = t - (h * 3600)
        m = int(float(th)/60.)
        s = th - (m * 60)
        if String:
            return '{}h, {}m, {}s'.format(h, m, s)
        else:
            if Round:
                return np.asarray((h, m, s), dtype=int)
            else:
                return np.asarray((h, m, s), dtype=float)

def Gaussian(x, mu=0, sig=1, norm=1):
    '''
    Standard 1D Gaussian function.
    '''
    var = sig ** 2
    height = norm / np.sqrt(2.*np.pi*var)
    funcval = height * np.exp(-(x-mu)**2/(2.*var))
    return funcval

def Gaussian2D(x, y, mu_x=0, mu_y=0, sig_x=1, sig_y=1, theta=0, norm=1):
    '''
    2D Gaussian function.
    '''
    A = 0.5 * ((np.cos(theta)/sig_x)**2 + (np.sin(theta)/sig_y)**2)
    B = 0.25 * np.sin(2*theta) * (sig_y**-2 - sig_x**-2)
    C = 0.5 * ((np.sin(theta)/sig_x)**2 + (np.cos(theta)/sig_y)**2)
    if isinstance(x, collections.Sized) and isinstance(y, collections.Sized):
        x, y = np.meshgrid(x, y)
    D = A * (x - mu_x)**2 + 2*B*(x - mu_x)*(y - mu_y) + C * (y - mu_y)**2
    height = norm / (2.*np.pi*sig_x*sig_y)
    funcval = height * np.exp(-D)
    return funcval

def logGaussian(x, mu=0, sig=1, norm=1):
    '''
    Standard 1D log-Gaussian function.
    '''
    if x == 0:
        return 0
    else:
        x = abs(x)
        var = sig ** 2
        height = norm / np.sqrt(2.*np.pi*var*x*x)
        funcval = height * np.exp(-(np.log(x)-mu)**2/(2.*var))
        return funcval

def npzcontents(npz, savetxt=None):
    '''
    For an already loaded npz archive file, this prints details of all files in the archive, as in npzload.
    '''
    totalbytes = 0
    totalfields = 0
    files = sorted(npz.files, key=str.casefold)
    if savetxt is None:
        print('Field   Shape   Bytes   Type')
        for f in files:
            npzf = npz[f]
            npzft = str(npzf.dtype)
            byte = sys.getsizeof(npzf)
            print(f, np.shape(npzf), IntWithCommas(byte), npzft)
            totalbytes += byte
            totalfields += 1
        print('Total Bytes: {}'.format(IntWithCommas(totalbytes)))
        print('Total Fields: {}'.format(IntWithCommas(totalfields)))
    elif savetxt in forbidden_names:
        raise Exception("'{}.txt' is already a file in the repo. Name this file something else.".format(savetxt))
    else:
        with open('{}.txt'.format(savetxt), 'w') as F:
            print('Field   Shape   Bytes   Type', file=F)
            for f in files:
                npzf = npz[f]
                npzft = str(npzf.dtype)
                byte = sys.getsizeof(npzf)
                print(f, np.shape(npzf), IntWithCommas(byte), npzft, file=F)
                totalbytes += byte
                totalfields += 1
            print('Total Bytes: {}'.format(IntWithCommas(totalbytes)), file=F)
            print('Total Fields: {}'.format(IntWithCommas(totalfields)), file=F)

def npzload(pathtonpz, savetxt=None):
    '''
    For an npz archive file with specified path, this reads the npz file while printing details of all files in the archive. To load the file without this output, simply use the numpy.load function.
    '''
    npz = np.load(pathtonpz)
    npzcontents(npz, savetxt=savetxt)
    return npz

def h5contents(h5, savetxt=None):
    '''
    For an already loaded hdf5 archive file, this prints details of all files in the archive, as in h5load.
    '''
    totalbytes = 0
    totalfields = 0
    keys = sorted(h5.keys(), key=str.casefold)
    if savetxt is None:
        print('Field   Shape   Bytes   Type')
        for f in keys:
            h5f = h5[f]
            h5ft = str(h5f.dtype)
            byte = sys.getsizeof(h5f)
            print(f, h5f.shape, IntWithCommas(byte), h5ft)
            totalbytes += byte
            totalfields += 1
        print('Total Bytes: {}'.format(IntWithCommas(totalbytes)))
        print('Total Fields: {}'.format(IntWithCommas(totalfields)))
    elif savetxt in forbidden_names:
        raise Exception("'{}.txt' is already a file in the repo. Name this file something else.".format(savetxt))
    else:
        with open('{}.txt'.format(savetxt), 'w') as F:
            print('Field   Shape   Bytes   Type', file=F)
            for f in keys:
                h5f = h5[f]
                h5ft = str(h5f.dtype)
                byte = sys.getsizeof(h5f)
                print(f, h5f.shape, IntWithCommas(byte), h5ft, file=F)
                totalbytes += byte
                totalfields += 1
            print('Total Bytes: {}'.format(IntWithCommas(totalbytes)), file=F)
            print('Total Fields: {}'.format(IntWithCommas(totalfields)), file=F)

def h5load(pathtoh5, savetxt=None):
    '''
    For an hdf5 archive file with specified path, this reads the hdf5 file while printing details of all files in the archive. To load the file without this output, simply use the h5py.File function.
    '''
    h5 = h5py.File(pathtoh5, 'r')
    h5contents(h5, savetxt=savetxt)
    return h5

hdf5load = h5load
hdf5contents = h5contents

def npz_equal(npz1, npz2):
    '''
    Shows whether any two npz files are identical. Arguments can be preloaded archives or paths to files.
    '''
    o = 0
    if isinstance(npz1, str):
        npz1 = np.load(npz1)
    if isinstance(npz2, str):
        npz2 = np.load(npz2)
    
    files1 = npz1.files
    files2 = npz2.files
    
    diff1 = sorted(list(set(files1) - set(files2)))
    diff2 = sorted(list(set(files2) - set(files1)))
    common = sorted(list(set(files1).intersection(files2)))
    
    for i in diff1:
        print('{} in npz1, not npz2'.format(i))
    if len(diff1) > 0:
        print('')
    
    for i in diff2:
        print('{} in npz2, not npz1'.format(i))
    if len(diff2) > 0:
        print('')
    
    for i in common:
        data1 = npz1[i]
        data2 = npz2[i]
        if np.array_equal(data1, data2) == False:
            o += 1
            print('{} Non-Identical.'.format(i))
    
    O = max([o, len(diff1), len(diff2)])
    if O == 0:
        print('NPZ Files Are Identical.')
        return True
    else:
        return False

def h5_equal(hdf1, hdf2):
    '''
    Shows whether any two hdf5 files are identical. Arguments can be preloaded archives or paths to files.
    '''
    o = 0
    if isinstance(hdf1, str):
        hdf1 = h5py.File(hdf1, 'r')
    if isinstance(hdf2, str):
        hdf2 = h5py.File(hdf2, 'r')
    
    files1 = hdf1.keys()
    files2 = hdf2.keys()
    
    diff1 = sorted(list(set(files1) - set(files2)))
    diff2 = sorted(list(set(files2) - set(files1)))
    common = sorted(list(set(files1).intersection(files2)))
    
    for i in diff1:
        print('{} in hdf1, not hdf2'.format(i))
    if len(diff1) > 0:
        print('')
    
    for i in diff2:
        print('{} in hdf2, not hdf1'.format(i))
    if len(diff2) > 0:
        print('')
    
    for i in common:
        data1 = hdf1[i]
        data2 = hdf2[i]
        if np.array_equal(data1, data2) == False:
            o += 1
            print('{} Non-Identical.'.format(i))
    
    O = max([o, len(diff1), len(diff2)])
    if O == 0:
        print('HDF5 Files Are Identical.')
        return True
    else:
        return False

hdf5_equal = h5_equal

def polyexp(X, params, base=10):
    '''
    Computes the exponential of a polynomial function of any order N.
    '''
    par = params[::-1]
    return np.exp(sum([par[i]*X**i for i in range(len(par))]) * np.log(base))

def cutdomain(arr):
    if len(arr) > 99:
        return arr[-99:]
    else:
        return arr

def relu(X):
    if X < 0:
        return 0
    else:
        return X

def find_nearest(array, value):
    '''
    For a given 1D array, and a number or set of numbers, this finds the element which is closest to each number, and the corresponding index of the array.
    '''
    array = np.asarray(array)
    if isinstance(value, collections.Sized) == False:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.asarray([(np.abs(array - Value)).argmin() for Value in value])
    return idx, array[idx]

def find_perc(array, value):
    '''
    For a given 1D array, and a percentile or set of percentiles, this finds the element which is closest to each percentile, and the corresponding index of the array.
    '''
    array = np.asarray(array)
    if isinstance(value, collections.Sized) == False:
        percval = np.percentile(array, value)
        idx = (np.abs(array - percval)).argmin()
    else:
        percval = np.percentile(array, value)
        idx = np.asarray([(np.abs(array - Value)).argmin() for Value in percval])
    return idx, array[idx]

def running_stdev(dat, window):
    '''
    Computes a standard deviation filter of a 1D dataset. Useful for computing scatter.
    '''
    Dat = np.zeros(len(dat))
    for i in range(1, len(Dat)-1):
        amin = int(max([0, i-0.5*window]))
        amax = int(min([len(Dat), i+0.5*window]))
        Dat[i] += np.std(dat[amin:amax])
    Dat[0] = Dat[1]
    Dat[-1] = Dat[-2]
    return Dat

def modelB(x, t=0):
    '''
    Computes the factor of dust attenuation from Nelson et al.: https://arxiv.org/abs/1707.03395/
    '''
    tau = (x / 5500)**-0.7
    if t > 1e-2:
        tau *= 0.3
    return np.exp(-tau)

def sort_arrays(arrays):
    '''
    For a tuple of at least two 1D arrays of equal length, this sorts all arrays with respect to the first array of the tuple.
    '''
    arrs = zip(*sorted(zip(*arrays)))
    Arrs = np.asarray([np.asarray(arr) for arr in arrs])
    return Arrs

TNGz, TNGa, TNGt, TNGH = sort_arrays((TNGredshift, TNGscalefactor, TNGtime, TNGHubble))

def count1D(array, order=True):
    cw = []
    if len(np.shape(array)) > 1:
        array = np.asarray(array).ravel()
    ck = Counter(array).keys()
    cv = Counter(array).values()
    if order:
        ck, cv = sort_arrays((ck, cv))
    else:
        cvr = max(cv) - cv
        cvr, ck, cv = sort_arrays((cvr, ck, cv))
    for cki in ck:
        cwi = np.where(array==cki)[0]
        cw.append(cwi)
    return ck, np.asarray(cv, dtype=int), tuple(cw)

def count(array, order=True, flat=False):
    '''
    For any array, this returns the elements with unique values, the frequency of each value in the array, and the indices at which they are found.
    '''
    arrshape = np.shape(array)
    if len(arrshape) == 1 or flat == True:
        return count1D(array, order=order)
    else:
        arrflat = array.ravel()
        ck = Counter(arrflat).keys()
        cv = Counter(arrflat).values()
        cw = []
        if order:
            ck, cv = sort_arrays((ck, cv))
        else:
            cvr = max(cv) - cv
            cvr, ck, cv = sort_arrays((cvr, ck, cv))
        for cki in ck:
            cwi = np.where(array==cki)
            cw.append(cwi)
        return ck, np.asarray(cv, dtype=int), tuple(cw)

def binmerger(data, bins, threshold=40):
    '''
    For a 1D dataset and a set of bin boundaries containing it, this function concatenates any neighbouring bins whose occupancy is smaller than a given threshold, and returns the boundaries of the new set of bins. The result will be a subset of the input bin arrays, which satisfies the condition of minumum occupancy.
    '''
    bins[-1] *= 1.01
    bins[0] *= 0.99
    digits = np.digitize(data, bins)
    for i in np.unique(digits):
        w = np.where(digits==i)[0]
        if len(w) <= threshold:
            for wi in w:
                digits[wi] += 1
    digits.clip(0, bins.size)
    newbins = [bins[0], bins[-1]]
    for i in np.unique(digits)[0:-1]:
        newbins.append(bins[i])
    return np.sort(newbins)

def midpoints(data, log=False):
    '''
    Returns the midpoints between every two consecutive points of the input array.
    '''
    if log:
        data = np.log(data)
        return np.exp([np.mean(data[i:i+2]) for i in range(len(data)-1)])
    else:
        return np.asarray([np.mean(data[i:i+2]) for i in range(len(data)-1)])

def rebindigits(oldbins, newbins):
    '''
    For two sets of bin boundaries, this returns the indices of the second set of bins which fall into each bin in the first.
    Note that a zero is appended to the first array, such that the first set of indices shows data below the lower bound. The output will have an additional element if any of the second bins are above the upper bound.
    '''
    assert min(oldbins) >= 0 and min(np.diff(oldbins)) > 0, "oldbins must be positive and monotonically increasing"
    assert min(newbins) >= 0 and min(np.diff(newbins)) > 0, "newbins must be positive and monotonically increasing"
    if min(oldbins) > 0:
        oldbins = np.append([0], oldbins)
    oldbins[-1] += 0.01
    means = midpoints(newbins)
    digits = np.digitize(means, oldbins)
    ck, cv, cw = count(digits)
    Ck = np.arange(min(ck), max(ck)+1)
    if np.array_equal(Ck, ck) == False:
        Cw = []
        for Cki in Ck:
            if Cki not in ck:
                Cw.append(np.asarray([]))
            else:
                j = np.where(ck==Cki)[0][0]
                Cw.append(cw[j])
        cw = tuple(Cw)
    return cw

def rebindigits_linear(oldbins, number):
    '''
    Special case of rebindigits, in which the second set of bins are linearly spaced between the bounds of the first set. The number of bins in the second set is specified and their boundaies and size are returned along with the bin indices.
    '''
    assert min(oldbins) >= 0 and min(np.diff(oldbins)) > 0, "oldbins must be positive and monotonically increasing"
    if min(oldbins) > 0:
        oldbins = np.append([0], oldbins)
    oldbins[-1] += 0.01
    newbins = np.linspace(min(oldbins), max(oldbins), number+1)
    means = midpoints(newbins)
    interval = means[1] - means[0]
    return newbins, interval, rebindigits(oldbins, newbins)

def FourierAmp(x, t=None, nfreq=None, log=None):
    '''
    Computes the Fourier amplitude of a real-valued 1D signal, optionally with a specified time domain.
    '''
    if t is None:
        t = np.linspace(0, 1, len(x))
    if nfreq is None:
        nfreq = len(x)
    
    t, x = sort_arrays((t, x))
    T = np.linspace(min(t), max(t), len(t))
    d = T[1] - T[0]
    nyfreq = 0.5 / d
    if np.array_equal(t, T) == False:
        x = np.interp(T, t, x)
    
    X = np.fft.rfft(x[x>0])
    freq = np.fft.rfftfreq(len(x[x>0]), d=d)
    Freq = np.linspace(min(freq), max(freq), nfreq)
    Y = np.interp(Freq, freq, np.abs(X))
    
    if log:
        return Freq, np.log(Y)/np.log(log), nyfreq
    else:
        return Freq, Y, nyfreq

def FourierAmps(x, t=None, nfreq=None, log=None):
    '''
    Computes the Fourier amplitude of a set of real-valued 1D signals, optionally with a specified time domain.
    '''
    FA = []
    for i in x:
        Freq, FT, nyfreq = FourierAmp(i, t=t, nfreq=nfreq, log=log)
        FA.append(FT)
    return Freq, np.asarray(FA), nyfreq

def psi(tng1, tng2, mh1, mh2, time=TNGtime, nbin=100):
    mh = np.concatenate((mh1, mh2))
    mhbins = np.percentile(mh, np.linspace(0, 100, nbin+1))
    dig1 = rebindigits(mhbins, mh1)
    dig2 = rebindigits(mhbins, mh2)
    mumh1 = np.asarray([np.mean(mh1[o]) for o in dig1])
    mumh2 = np.asarray([np.mean(mh2[o]) for o in dig2])
    Psi = []
    for o in range(len(dig1)):
        h1 = tng1[dig1[o]]
        h2 = tng2[dig2[o]]
        h1mask = np.ma.masked_where(h1 == 0, h1)
        h2mask = np.ma.masked_where(h2 == 0, h2)
        h1mu = np.ma.mean(h1mask, axis=0).filled(1)
        h2mu = np.ma.mean(h2mask, axis=0).filled(1)
        Psio = h1mu / h2mu
        Psi[Psio>100] = 1
        Psi.append(Psio)
    Psi = np.asarray(Psi)
    Psi = np.clip(Psi, a_min=1, a_max=100)
    Interp = interp2d(time, mumh1, Psi)
    return mumh1, Psi, Interp

ugriz = 'ugriz'
sdssfilters = np.load('sdssfilters.npy')
fwave = sdssfilters[0]
bandpass = sdssfilters[1:]

def IsNDArray(array, N):
    '''
    Determines whether an array has a given number of dimensions.
    '''
    if len(np.shape(array)) == N:
        return True
    else:
        return False

def sfhint(sfh, time, log=False):
    '''
    Computes an integral of star formation histories over cosmic time to obtain a final (unrecycled) stellar mass value.
    '''
    sfh_int = []
    X = np.append([0], time)
    if IsNDArray(sfh, 2):
        assert len(sfh[0]) == len(time), "len(time) must match axis 1 of SFH."
        for i in sfh:
            Y = np.append([0], i)
            ms = np.trapz(Y, x=X)
            sfh_int.append(ms)
        if log:
            return np.log10(sfh_int)
        else:
            return np.asarray(sfh_int)
    elif IsNDArray(sfh, 1):
        assert np.shape(time) == np.shape(sfh), "SFH must be same length as time domain."
        Y = np.append([0], sfh)
        ms = np.trapz(Y, x=X)
        if log:
            return np.log10(ms)
        else:
            return ms
    else:
        raise Exception: "SFH must be a 1D or 2D array, of shape (n_samples, n_timesteps)."

def mwz(zh, sfh, time, log=False):
    '''
    Computes the mass weighted metallicity of an input galaxy or set of galaxies.
    '''
    assert np.shape(zh) == np.shape(sfh), "SFH and ZH must be same shape."
    dt = np.diff(np.append([min(TNGtime)], np.sort(time)))
    if IsNDArray(zh, 2):
        MWZ = []
        assert len(sfh[0]) == len(time), "len(time) must match axis 1 of SFH/ZH."
        for i in range(len(sfh)):
            ti, sfhi, zhi = sort_arrays((time, sfh[i], zh[i]))
            mwzi = np.average(zhi, weights=sfhi*dt)
            MWZ.append(mwzi)
        if log:
            return np.log10(MWZ)
        else:
            return np.asarray(MWZ)
    elif IsNDArray(zh, 1):
        assert np.shape(time) == np.shape(sfh), "SFH and ZH must be same length as time domain."
        ti, sfhi, zhi = sort_arrays((time, sfh, zh))
        MWZ = np.average(zhi, weights=sfhi*dt)
        if log:
            return np.log10(MWZ)
        else:
            return np.asarray(MWZ)
    else:
        raise Exception: "SFH and ZH must be 1D or 2D arrays, of equal shape (n_samples, n_timesteps)."

def mwa(lookback, sfh):
    '''
    Computes the mass weighted age(s) of input SFH(s).
    '''
    MWA = []
    if IsNDArray(sfh, 2):
        assert len(sfh[0]) == len(lookback), "len(lookback) must match axis 1 of SFH."
        for i in sfh:
            mwai = np.average(lookback, weights=i)
            MWA.append(mwai)
        return np.asarray(MWA)
    elif IsNDArray(sfh, 1):
        assert np.shape(lookback) == np.shape(sfh), "SFH must be same length as lookback time domain."
        return np.average(lookback, weights=sfh)
    else:
        raise Exception: "SFH must be a 1D or 2D array, of shape (n_samples, n_timesteps)."

def ABmag0(x, y, Band, D=10, ModelB=False, flambda=True, spline='linear', Nint=100000, xlog=False, ylog=False):
    assert len(x) == len(y), "x and y must be equal in size."
    if xlog:
        x = 10**x
    if ylog:
        y = 10**y
    x, y = sort_arrays((x, y))
    D = 0.1 * D
    if flambda:
        y = y * x**2 * 3.34e4 / 3.827e29 / 3631
    else:
        y = y * 7.2107927e-11
    band = {'u':bandpass[0], 'g':bandpass[1], 'r':bandpass[2], 'i':bandpass[3], 'z':bandpass[4]}
    try:
        band_pass = band[Band]
    except KeyError:
        raise Exception('Valid Bands: u, g, r, i, z')
    f1 = interp1d(x, y, bounds_error=False, fill_value=0, kind=spline)
    f2 = interp1d(fwave, band_pass, bounds_error=False, fill_value=0, kind=spline)
    X = np.linspace(min([min(x), min(fwave)]), max([max(x), max(fwave)]), int(Nint))
    Y = f1(X) * f2(X)
    if ModelB:
        Y = Y * modelB(X)
    Flux = np.trapz(Y, x=X) / D**2
    FluxMag = -2.5*np.log10(Flux) - 48.6
    return FluxMag

def ABmag(x, y, Band, D=10, ModelB=False, flambda=True, spline='linear', Nint=100000, xlog=False, ylog=False):
    '''
    Computes AB band magnitudes by integrating a converted spectrum over a specified bandpass filter. Optionally applies simple dust attenuation and luminosity distance. Supports the five SDSS `ugriz' bandpasses.
    Input spectrum flux density converted to maggies, for compatibility with the AB magnitude system.
    '''
    bandsized = isinstance(Band, list)
    y2D = IsNDArray(y, 2)
    if bandsized:
        if y2D:
            abmag = []
            for yi in y:
                abmag.append([ABmag0(x, yi, Bi, D=D, ModelB=ModelB, flambda=flambda, spline=spline, Nint=Nint, xlog=xlog, ylog=ylog) for Bi in Band])
        else:
            abmag = [ABmag0(x, y, Bi, D=D, ModelB=ModelB, flambda=flambda, spline=spline, Nint=Nint, xlog=xlog, ylog=ylog) for Bi in Band]
    else:
        if y2D:
            abmag = [ABmag0(x, yi, Band, D=D, ModelB=ModelB, flambda=flambda, spline=spline, Nint=Nint, xlog=xlog, ylog=ylog) for yi in y]
        else:
            return ABmag0(x, y, Band, D=D, ModelB=ModelB, flambda=flambda, spline=spline, Nint=Nint, xlog=xlog, ylog=ylog)
    return np.asarray(abmag)
