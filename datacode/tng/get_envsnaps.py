from basemodule import *
from tngmodule import *

# This code computes and saves the mass, radius and location of all subhalos at every snapshot in the four main simulations.

a = subhistory(simulation=100, order=1, Dark=False) # TNG100-1
A = subhistory(simulation=300, order=1, Dark=False) # TNG300-1
b = subhistory(simulation=100, order=1, Dark=True) # TNG100-1-Dark
B = subhistory(simulation=300, order=1, Dark=True) # TNG300-1-Dark

# For each snapshot, an npz file is saved in the envsnaps directory if baryonic, envsnaps-dark if dark. Filename indicates simulation.
# These directories are created if they do not already exist in the current working directory.

# N.B. The total size of these files is large (approx. 20GB per simulation). The TNG team may request that you delete them from the TNG server.
