from basemodule import *
import illustris_python as il

redshift=0
scale_factor = 1.0 / (1+redshift)
h = 0.6774
H0 = h * 100
solar_Z = 0.0127
Om0 = 0.3089
OL0 = 1 - Om0
convfactor = 1.0e10 / 0.6774
rconvfactor = 1.0 / (0.6774 * 1e3)
cosmodata = np.genfromtxt('cosmo.txt')
lookback_time = us(cosmodata[0], cosmodata[1], s=0)
simdims = {50:35/0.6774, 100:75/0.6774, 300:205/0.6774}

TNGbins = np.append(0, TNGtime)
TNGbins[-1] += 0.01
Nt=10000

def sfhzh(index, simulation=100, order=1, Dark=False, snapnum=99, time=TNGtime, NT=10000, radius=None):
	'''
	Returns the SFH and ZH of a subhalo from the TNG database, with custom binning and optional stellar particle aperture.
	'''
	sfh_t_boundary, lt, cw = rebindigits_linear(time, NT)
	if Dark:
		basePath = 'sims.TNG/TNG{}-{}-Dark/output/'.format(simulation, order)
	else:
		basePath = 'sims.TNG/TNG{}-{}/output/'.format(simulation, order)
	reqfields=['GFM_InitialMass','GFM_Metallicity','GFM_StellarFormationTime','Coordinates']
	s0=il.snapshot.loadSubhalo(basePath, snapnum, index, 'stars', fields=reqfields)
	a_formation = s0['GFM_StellarFormationTime'][:]
	metals = s0['GFM_Metallicity'][:]
	mass = s0['GFM_InitialMass'][:]
	if radius is None:
		w = np.where((a_formation>0))[0]
	else:
		rx, ry, rz = s0['Coordinates'].T * rconvfactor
		Rxyz = np.asarray([np.average(rx, weights=mass), np.average(ry, weights=mass), np.average(rz, weights=mass)])
		rr = distance(np.asarray([rx, ry, rz]), Rxyz, simdims[int(simulation)])
		w = np.where((a_formation>0) & (rr < radius))[0]
	z_formation = 1.0/a_formation[w] - 1
	age = lookback_time(z_formation)
	mass = mass[w] * convfactor
	metals = metals[w] / solar_Z
	sfh, x = np.histogram(age, bins=sfh_t_boundary, weights=mass)
	z, x = np.histogram(age, bins=sfh_t_boundary, weights=mass*metals)
	zh = np.zeros(NT)
	zh[sfh>0] = z[sfh>0] / sfh[sfh>0]
	sfh = sfh[::-1] / lt
	zh = zh[::-1]
	sfp = np.asarray([np.mean(sfh[i]) for i in cw])
	zp = np.asarray([np.average(zh[i], weights=sfh[i]) for i in cw])
	return sfp, rmNanInf(zp)

def subhistory(simulation=100, order=1, Dark=False):
	'''
	Computes the mass, radius and location histories of all subhalos in the chosen simulation run. These are saved to 100 npz files; one for each snapshot number.
	This algorithm can take time to complete, and will save a few dozen gigabytes per simulation run. Files are saved in the directory envsnaps, which is created if it does not exist.
	This data is necessary for running functions in the GriSPy Module.
	'''
	if Dark:
		nudir = 'envsnaps-dark'
	else:
		nudir = 'envsnaps'
	if not os.path.exists(nudir):
		os.makedirs(nudir)
	if Dark:
		bP = 'sims.TNG/TNG{}-{}-Dark/output'.format(simulation, order)
	else:
		bP = 'sims.TNG/TNG{}-{}/output'.format(simulation, order)
	if order > 1:
		simulation = '{}-{}'.format(simulation, order)
	for i in range(100):
		CM = il.groupcat.loadSubhalos(bP, i, fields=['SubhaloCM']) * rconvfactor
		MH = il.groupcat.loadSubhalos(bP, i, fields=['SubhaloMassType'])[:,1] * convfactor
		R = il.groupcat.loadSubhalos(bP, i, fields=['SubhaloHalfmassRadType'])[:,1] * rconvfactor
		H = il.groupcat.loadSubhalos(bP, i, fields=['SubhaloGrNr'])
		np.savez('{}/tng{}s{}_sub.npz'.format(nudir, simulation, i), Parent=H, Location=CM, MSub=MH, RSub=R)
		print('TNG{} Subhistory: {}% Complete.'.format(simulation, i+1))

def IsCent(subIDs, snapnum=99, simulation=100, order=1, Dark=False):
	'''
	States whether given SubfindID(s) are those of a central subhalo or not.
	'''
	if Dark:
		bP = 'sims.TNG/TNG{}-{}-Dark/output'.format(simulation, order)
	else:
		bP = 'sims.TNG/TNG{}-{}/output'.format(simulation, order)
	centIDs = il.groupcat.loadHalos(bP, snapnum, fields=['GroupFirstSub'])
	isin = np.in1d(subIDs, centIDs)
	if len(isin) > 1:
		return isin
	else:
		return isin[0]

def hostIDs(subIDs, snapnum=99, simulation=100, order=1, Dark=False):
	'''
	For SubfindID(s), returns the FoF ID of its host(s), and the SubfindID(s) of the central subhalo(s). States whether the given SubfindID is the central subhalo or not.
	'''
	if Dark:
		bP = 'sims.TNG/TNG{}-{}-Dark/output'.format(simulation, order)
	else:
		bP = 'sims.TNG/TNG{}-{}/output'.format(simulation, order)
	haloIDs = il.groupcat.loadSubhalos(bP, snapnum, fields=['SubhaloGrNr'])
	haloIDs = haloIDs[subIDs]
	centIDs = il.groupcat.loadHalos(bP, snapnum, fields=['GroupFirstSub'])
	iscent = np.in1d(subIDs, centIDs)
	centIDs = centIDs[haloIDs]
	return haloIDs, centIDs, iscent

def Infall(subIDs, snapnum=99, simulation=100, order=1, Dark=False, save=None):
	'''
	Computes an array of infall parameters for a set of subhalo IDs from a given simulation run.
	'''
	if Dark:
		bP = 'sims.TNG/TNG{}-{}-Dark/output'.format(simulation, order)
	else:
		bP = 'sims.TNG/TNG{}-{}/output'.format(simulation, order)
	cfields = ['SubhaloGrNr', 'GroupVel', 'GroupMassType']
	sfields = ['SubhaloGrNr', 'SubhaloVel', 'SubhaloMassType']
	Mu, aInfall, aMax, Vel, MH, mH, MS, mS, MG, mG = [[] for i in range(10)]
	haloIDs, centIDs, iscent = hostIDs(subIDs, snapnum=snapnum, simulation=simulation, order=order)
	Sim = [int(simulation) for i in range(len(subIDs))]
	for i in range(len(subIDs)):
		sid = subIDs[i]
		cid = centIDs[i]
		tree = il.sublink.loadTree(bP, snapnum, sid, fields=sfields, onlyMPB=True)
		Tree = il.sublink.loadTree(bP, snapnum, cid, fields=cfields, onlyMPB=True)
		g = tree['SubhaloGrNr']
		G = Tree['SubhaloGrNr']
		gG = np.asarray([abs(g[j] - G[j]) for j in range(min((len(G), len(g))))])
		if max(gG) > 0:
			j = min(np.where(gG!=0)[0])
		else:
			j = len(gG) - 1
		zj = TNGz[j]
		v = tree['SubhaloVel'][j]
		V = Tree['GroupVel'][j] * TNGa[j]
		vel = np.log10(np.sqrt(np.sum((v - V)**2)))
		mh = tree['SubhaloMassType'][0:len(gG),1]
		Mh = Tree['GroupMassType'][0:len(gG),1]
		ms = tree['SubhaloMassType'][0:len(gG),4]
		Ms = Tree['GroupMassType'][0:len(gG),4]
		mg = tree['SubhaloMassType'][0:len(gG),0]
		Mg = Tree['GroupMassType'][0:len(gG),0]
		halfmaxmh = max(mh) * 0.5
		mhc = np.maximum.accumulate(mh[::-1])[::-1]
		J1 = np.argmax(mh)
		J2 = halfmax(mhc, halfmaxmh)
		aM = TNGa[J1] / TNGa[J2]
		aI = TNGa[j] / TNGa[J2]
		m = mh[j]
		M = Mh[j]
		MH.append(M)
		mH.append(m)
		if Dark == False:
			MS.append(Ms[j])
			mS.append(ms[j])
			MG.append(Mg[j])
			mG.append(mg[j])
		mu = np.log10(m / M)
		Mu.append(mu)
		aInfall.append(aI)
		aMax.append(aM)
		Vel.append(vel)
	if Dark:
		InfallParams = np.asarray([Sim, aInfall, aMax, Mu, Vel, MH, mH, subIDs, haloIDs, centIDs])
	else:
		InfallParams = np.asarray([Sim, aInfall, aMax, Mu, Vel, MH, mH, MS, mS, MG, mG, subIDs, haloIDs, centIDs])
    if save is not None:
		if Dark:
			np.save('{}-Dark.npy'.format(save), InfallParams)
		else:
			np.save('{}.npy'.format(save), InfallParams)
	return InfallParams
