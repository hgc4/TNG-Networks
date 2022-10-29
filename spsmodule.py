from basemodule import *
import fsps

RecycleFraction = 0.43
h = 0.6774
t_BC = 0.03
F = 1e10 / (h * (1 - RecycleFraction))
Z0 = 0.0127
L = 3.828e26
Time = []

fluxlines = {}
flux_lines = {}
with open("fluxlines.txt") as f:
	for line in f:
		key, val = line.split()
		fluxlines[key] = int(val)
		flux_lines[int(val)] = key

ssp = fsps.StellarPopulation(zcontinuous=1,logzsol=0,sfh=0,imf_type=1,add_neb_emission=True,nebemlineinspec=True)
wave, flux = ssp.get_spectrum(tage=1, peraa=True)
linewave = ssp.emline_wavelengths
wave = np.asarray(wave)
D = len(wave)

def waverange(lambdamin=2980, lambdamax=11230):
	'''
	With specified lower and upper wavelengths, this returns the location indices of the FSPS wavelength array which contains them.
	'''
	if lambdamax < lambdamin:
		lambdamax, lambdamin = lambdamin, lambdamax
	w = np.where((wave<=lambdamax) & (wave>=lambdamin))[0]
	if len(w) == 0:
		index, placeholder = find_nearest(wave, np.mean([lambdamin, lambdamax]))
		return np.arange(index-1, index+2, dtype=int)
	else:
		return np.concatenate(([max([min(w)-1, 0])], w, [min([D-1, 1+max(w)])]))

def spectrum(sfh, zh, lookback, pera=True, lambdamin=2980, lambdamax=11230):
	'''
	Computes the spectral energy distribution from a given star formation and metallicity history.
	'''
	assert len(sfh) == len(zh) and len(sfh) == len(lookback), "sfh, zh and lookback time must be same length"
	lookback, sfh, zh = sort_arrays((lookback, sfh, zh))
	lookbacktime = np.append(lookback, [max(TNGtime)])
	deltat = np.abs(np.diff(lookbacktime))
	Spectrum = np.zeros(D)
	for j in range(len(sfh)):
		if sfh[j] > 0:
			ssp.params['logzsol'] = np.log10(zh[j])
			wave, flux = ssp.get_spectrum(tage=max([lookback[j], 0.01]), peraa=pera)
			Spectrum = Spectrum + np.array(flux) * sfh[j] * deltat[j]
	wave = wave[waverange(lambdamin, lambdamax)]
	Spectrum = Spectrum[waverange(lambdamin, lambdamax)]
	return wave, Spectrum

def spectra(sfh, zh, lookback, pera=True, lambdamin=2980, lambdamax=11230, verbose=True, numperc=100, string='Spectra'):
	'''
	Computes spectral energy distributions as in spectrum, for 2D arrays of multiple star formation and metallicity histories.
	As this can take time with large datasets, this includes support for the verb function to print progress updates.
	'''
	assert len(sfh[0]) == len(zh[0]) and len(sfh[0]) == len(lookback), "sfh, zh and lookback time must be same length on time axis"
	assert np.shape(sfh) == np.shape(zh), "sfh and zh must have identical dimensions"
	Spectra = []
	for i in range(len(sfh)):
		wave, Spectrum = spectrum(sfh[i], zh[i], lookback=lookback, pera=pera, lambdamin=lambdamin, lambdamax=lambdamax)
		Spectra.append(Spectrum)
		if verbose:
			verb(i, len(sfh), numperc=numperc, string=string)
	return wave, np.asarray(Spectra)

def linewavelength(line='Halpha'):
	'''
	Returns the wavelength of an emission line specified by a key.
	'''
	try:
		i = fluxlines[line]
	except KeyError:
		raise Exception("Line names are given in the format 'Halpha', 'Brdelta', etc. for hydrogen lines, and 'HeI_6680', 'CII_158mu', etc. for other elements.\nFor full list run linekeys().")
	return linewave[i]

def linekeys(savetxt=None):
	'''
	Prints the full, enumerated list of emission line keys for use in related functions, and their wavelengths, in ascending order of wavelength.
	'''
	if savetxt is None:
		for i in range(len(linewave)):
			j = flux_lines[i]
			k = linewavelength(j)
			print(i, j, k)
	elif savetxt in forbidden_names:
		raise Exception("'{}.txt' is already a file in the repo. Name this file something else.".format(savetxt))
	else:
		with open('{}.txt'.format(savetxt), 'w') as F:
			for i in range(len(linewave)):
				j = flux_lines[i]
				k = linewavelength(j)
				print(i, j, k, file=F)

def lineflux(sfh, zh, lookback, line='Halpha'):
	'''
	Computes the luminosity of an emission line for a given star formation and metallicity history.
	'''
	assert isinstance(line, str) or isinstance(line, list), "'line' argument must be string or list of strings.\nFor full list of valid keys, run linekeys()."
	assert len(sfh) == len(zh) and len(sfh) == len(lookback), "sfh, zh and lookback time must be same length"
	lookback, sfh, zh = sort_arrays((lookback, sfh, zh))
	lookbacktime = np.append(lookback, [max(TNGtime)])
	deltat = np.abs(np.diff(lookbacktime))
	if isinstance(line, str):
		Flux = 0
		try:
			i = fluxlines[line]
		except KeyError:
			raise Exception("Line names are given in the format 'Halpha', 'Brdelta', etc. for hydrogen lines, and 'HeI_6680', 'CII_158mu', etc. for other elements.\nFor full list run linekeys().")
		for j in range(len(sfh)):
			if sfh[j] > 0:
				ssp.params['logzsol'] = np.log10(zh[j])
				wave, flux = ssp.get_spectrum(tage=max([lookback[j], 0.01]), peraa=True)
				lines = ssp.emline_luminosity * sfh[j] * deltat[j]
				Flux += np.nan_to_num(lines[i])
	else:
		Flux = np.zeros(len(line))
		try:
			i = np.asarray([fluxlines[Line] for Line in line])
		except KeyError:
			raise Exception("Line names are given in the format 'Halpha', 'Brdelta', etc. for hydrogen lines, and 'HeI_6680', 'CII_158mu', etc. for other elements.\nFor full list run linekeys().")
		for j in range(len(sfh)):
			if sfh[j] > 0:
				ssp.params['logzsol'] = np.log10(zh[j])
				wave, flux = ssp.get_spectrum(tage=max([lookback[j], 0.01]), peraa=True)
				lines = ssp.emline_luminosity * sfh[j] * deltat[j]
				Flux += np.nan_to_num(lines[i])
	return np.nan_to_num(Flux)

def linefluxes(sfh, zh, lookback, line='Halpha', verbose=True, numperc=100, string='Line Fluxes'):
	'''
	Computes emission line luminosity as in lineflux, for 2D arrays of multiple star formation and metallicity histories.
	As this can take time with large datasets, this includes support for the verb function to print progress updates.
	'''
	assert len(sfh[0]) == len(zh[0]) and len(sfh[0]) == len(lookback), "sfh, zh and lookback time must be same length on time axis"
	assert np.shape(sfh) == np.shape(zh), "sfh and zh must have identical dimensions"
	Fluxes = []
	for i in range(len(sfh)):
		Flux = lineflux(sfh[i], zh[i], lookback=lookback, line=line)
		Fluxes.append(Flux)
		if verbose:
			verb(i, len(sfh), numperc=numperc, string=string)
	return np.asarray(Fluxes)
