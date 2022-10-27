from basemodule import *
from grispmodule import *

d1 = h5py.File('central_nndata_train.h5', 'r')
d2 = h5py.File('satellite_nndata_train.h5', 'r')

locC = d1['centreofmass'][:]
locS = d2['centreofmass'][:]

oC = Overdensities(locC, radius=np.asarray([1,3,5]))
oS = Overdensities(locS, radius=1)

np.save('densityC.npy', oC)
np.save('densityS.npy', oS)
