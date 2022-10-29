from basemodule import *
from grispy import GriSPy

L100 = 75/0.6774
L300 = 205/0.6774
V100 = L100**3
V300 = L300**3
P100 = {0: (0, L100), 1: (0, L100), 2: (0, L100)}
P300 = {0: (0, L300), 1: (0, L300), 2: (0, L300)}

loc100c = np.load('CM100_sub.npy')
loc300c = np.load('CM300_sub.npy')
loc100s = np.load('CM100Sat.npy')
loc300s = np.load('CM300Sat.npy')

def Overdensity(loc, radius=1, sim=100, snap=99, Dark=False):
	'''
	Computes the DM overdensity within spherical regions in TNG, specified by a list of coordinates and radii.
	'''
	vol = 4 * np.pi * radius ** 3 / 3
	sim = int(sim)
	validsims = [100, 300]
	assert sim in validsims, "sim must be 100 or 300"
	
	if Dark:
		data = np.load('envsnaps-dark/tng{}s{}_sub.npz'.format(sim, snap))
	else:
		data = np.load('envsnaps/tng{}s{}_sub.npz'.format(sim, snap))
	M = data['MSub']
	CM = data['Location']
	
	if float(sim) < 120:
		L = L100
		V = V100
		P = P100
	else:
		L = L300
		V = V300
		P = P300
	
	nz = np.where(M>0)[0]
	M = M[nz]
	CM = CM[nz]
	rhobar = sum(M) / V
	
	gsp = GriSPy(CM, periodic=P)
	
	if isinstance(radius, collections.Sized) == False:
		bdist, binds = gsp.bubble_neighbors(loc, distance_upper_bound=radius)
		MR = []
		for o in range(len(binds)):
			D0 = bdist[o]
			I0 = binds[o]
			MR.append(sum(M[I0[D0>0]]))
		rho = np.asarray(MR) / vol
	else:
		rho = []
		for j, i in enumerate(radius):
			bdist, binds = gsp.bubble_neighbors(loc, distance_upper_bound=i)
			MR = []
			for o in range(len(binds)):
				D0 = bdist[o]
				I0 = binds[o]
				MR.append(sum(M[I0[D0>0]]))
			rho.append(np.asarray(MR) / vol[j])
		rho = np.asarray(rho)
	delta = rho / rhobar
	return delta

def Overdensities(locs, radius=1, sim=100, Dark=False, verbose=True, numperc=100, string='Overdensity History'):
	'''
	Computes dark matter overdensities as in Overdensity, now evaluated at all snapshots in TNG, thereby computing a full overdensity history.
	'''
	delta = []
	for j in range(100):
		Delta = Overdensity(locs[:,j], radius=radius, sim=sim, snap=j, Dark=Dark)
		delta.append(Delta)
		if verbose:
			verb(j, 100, numperc=numperc, string=string)
	return np.asarray(delta)

def Skew(loc, radius=3, sim=100, snap=99, Dark=False):
	'''
	Computes the dark matter weighted radial skew within spherical regions in TNG, specified by a list of coordinates and radii.
	'''
	vol = 4 * np.pi * radius ** 3 / 3
	sim = int(sim)
	validsims = [100, 300]
	assert sim in validsims, "sim must be 100 or 300"
	
	if Dark:
		data = np.load('envsnaps-dark/tng{}s{}_sub.npz'.format(sim, snap))
	else:
		data = np.load('envsnaps/tng{}s{}_sub.npz'.format(sim, snap))
	M = data['MSub']
	CM = data['Location']
	
	if float(sim) < 120:
		L = L100
		V = V100
		P = P100
	else:
		L = L300
		V = V300
		P = P300
	
	nz = np.where(M>0)[0]
	M = M[nz]
	CM = CM[nz]
	muC = np.mean(loc, axis=1)
	
	gsp = GriSPy(CM, periodic=P)
	skewness, MinD = [[], []]
	
	if isinstance(radius, collections.Sized) == False:
		bdist, binds = gsp.bubble_neighbors(loc, distance_upper_bound=radius)
		MR = np.asarray([sum(M[bind]) for bind in binds])
		for o in range(len(MR)):
			D0 = bdist[o]
			M0 = MR[o]
			if len(D0) > 2 and muC[o] > 0:
				D0, M0 = sort_arrays((D0, M0))
				skew = n_weighted_moment(D0[D0>0], M0[D0>0], 3)
				mind = min(D0[1:])
				skewness.append(skew)
				MinD.append(mind)
			else:
				mind = -150
				skewness.append(mind)
				MinD.append(mind)
	else:
		Skewness, Min_D = [[], []]
		for i in radius:
			bdist, binds = gsp.bubble_neighbors(loc, distance_upper_bound=i)
			MR = np.asarray([sum(M[bind]) for bind in binds])
			for o in range(len(MR)):
				D0 = bdist[o]
				M0 = MR[o]
				if len(D0) > 2 and muC[o] > 0:
					D0, M0 = sort_arrays((D0, M0))
					skew = n_weighted_moment(D0[D0>0], M0[D0>0], 3)
					mind = min(D0[1:])
					Skewness.append(skew)
					Min_D.append(mind)
				else:
					mind = -150
					Skewness.append(mind)
					Min_D.append(mind)
		skewness.append(Skewness)
		MinD.append(Min_D)
	return np.asarray(skewness), np.asarray(MinD)

def Skews(locs, radius=3, sim=100, Dark=False, verbose=True, numperc=100, string='Skewness History'):
	'''
	Computes radial skews as in Skew, now evaluated at all snapshots in TNG, thereby computing a full skew history.
	'''
	mu3, dmu3 = [[], []]
	for j in range(100):
		Mu3, dMu3 = Skew(locs[:,j], radius, sim=sim, snap=j, Dark=Dark)
		mu3.append(Mu3)
		dmu3.append(dMu3)
		if verbose:
			verb(j, 100, numperc=numperc, string=string)
	return np.asarray(mu3), np.asarray(dmu3)
