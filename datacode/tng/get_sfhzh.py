from basemodule import *
from tngmodule import *

# This code computes and saves the star formation and metallicity histories of randomly selected subhalos in a chosen simulation.

# Choose the simulation run
simulation = 100 # must be 50, 100 or 300, in int format
order = 1 # must be a number from 1 to 4, inclusive, in int format
Dark = False # must be True or False

TNGbins = np.append(0, TNGtime)
TNGbins[-1] += 0.01
Nt=10000

IDs = np.unique(np.random.randint(low=0, high=100000, size=400)) # this can be any list of integers up to the total number of subhalos in the chosen snapshot

sfh, zh = [[], []]
for ID in IDs:
	SFH, ZH = sfhzh(ID, simulation=simulation, order=order, Dark=Dark, snapnum=99, time=TNGtime, NT=10000, radius=None) # computes SFH and ZH recursively
	sfh.append(SFH)
	zh.append(ZH)

if Dark:
	np.savez_compressed('SFH-ZH_TNG{}-{}-Dark.npz'.format(simulation, order), ID=IDs, SFH=sfh, ZH=zh) # saves SFH and ZH of all IDs to cwd
else:
	np.savez_compressed('SFH-ZH_TNG{}-{}.npz'.format(simulation, order), ID=IDs, SFH=sfh, ZH=zh)
