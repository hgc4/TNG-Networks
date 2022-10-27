from basemodule import *
from grispmodule import *

d1 = h5py.File('central_nndata_train.h5', 'r')
d2 = h5py.File('satellite_nndata_train.h5', 'r')

locC = d1['centreofmass'][:]
locS = d2['centreofmass'][:]

kC, mC = Skews(locC, radius=np.asarray([1,3,5]))
kS, mS = Skews(locS, radius=1)

np.save('skewC.npy', kC)
np.save('skewS.npy', kS)

np.save('dminC.npy', mC)
np.save('dminS.npy', mS)
