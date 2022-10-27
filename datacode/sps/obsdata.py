from basemodule import *
from spsmodule import *

d0 = h5py.File('central_nndata_train.h5', 'r')
SFH = d0['SFH'][:]
ZH = d0['ZH'][:]

lookback_time = max(TNGtime) - TNGtime
wave, Spectra = spectra(SFH, ZH, lookback_time)
Halpha = linefluxes(SFH, ZH, lookback_time)
Hbeta = linefluxes(SFH, ZH, lookback_time, line='Hbeta')
Mags = ABmag(wave, Spectra, ['u', 'g', 'r', 'i', 'z'])

np.savez_compressed('ObsData.npz', wavelength=wave, SED=Spectra, H_Alpha=Halpha, H_Beta=Hbeta, magnitude=Mags)
