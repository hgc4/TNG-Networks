from basemodule import *
from tngmodule import *

# This code computes and saves the infall parameters of randomly selected satellite subhalos in a chosen simulation.

# Choose the simulation run
simulation = 100 # must be 50, 100 or 300, in int format
order = 1 # must be a number from 1 to 4, inclusive, in int format
Dark = False # must be True or False

IDs = np.unique(np.random.randint(low=0, high=100000, size=400)) # this can be any list of integers up to the total number of subhalos in the chosen snapshot
Inf = Infall(IDs, snapnum=99, simulation=simulation, order=order, Dark=Dark, save='Infall_TNG{}-{}'.format(simulation, order)) # computes and saves infall parameters
