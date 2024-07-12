from basemodule import *
from nnmodule import *

d0 = h5py.File('central_nndata_train.h5', 'r')
d1 = h5py.File('central_nndata_test.h5', 'r')
d2 = h5py.File('central_nndata_pred.h5', 'r')
d3 = h5py.File('central_nndata_dark.h5', 'r')
model = keras.models.load_model('centralmodel.h5')
time = d3['time'][:]
age = d3['lookback_time'][:]

SFH0 = np.concatenate((d0['SFH'][:], d1['SFH'][:]), axis=0)
SFHq, SFHg = GQTvecnorm(SFH0.reshape(*np.shape(SFH0), 1))

ZH0 = np.concatenate((d0['ZH'][:], d1['ZH'][:]), axis=0)
ZHq, ZHg = GQTvecnorm(ZH0.reshape(*np.shape(ZH0), 1))

d3xt = np.stack((d3['Mhdot'][:], d3['delta1'][:], d3['delta3'][:], d3['delta5'][:], d3['vcirc'][:], d3['rhalf'][:], d3['skew'][:], d3['minD'][:]), axis=-1)
d3xs = np.stack((d3['beta'][:], d3['d_min'][:], d3['d_node'][:], d3['d_saddle_1'][:], d3['d_saddle_2'][:], d3['d_skel'][:], d3['formtime'][:], d3['logMh'][:], d3['logmaxMhdot'][:]), axis=-1)

d3xt1q, d3xt1g = GQTvecnorm(d3xt[:,:,0].reshape(len(d3xt), 33, 1))
d3xt2q, d3xt2g = GQTscalnorm(d3xt[:,:,1:4])
d3xt3q, d3xt3g = GQTvecnorm(d3xt[:,:,4:])
d3xtq = np.stack((d3xt1q, d3xt2q, d3xt3q), axis=-1)
d3xsq, d3xsg = GQTscalnorm(d3xs)
d3xq = [d3xtq, d3xsq]
d3yq = model.predict(d3xq)

SFH = np.squeeze(GQTvecinv(d3yq[:,0:33].reshape(len(d3xs), 33, 1), SFHg))
ZH = np.squeeze(GQTvecinv(d3yq[:,33:66].reshape(len(d3xs), 33, 1), ZHg))

X1 = 10**d1['logMh'][:]
X2 = 10**d2['logMh'][:]
X3 = 10**d3['logMh'][:]

M1 = d1['logMS'][:]
M2 = d2['logMS'][:]
M3 = sfhint(SFH, time=time)

Z1 = d1['logZS'][:]
Z2 = d2['logZS'][:]
Z3 = mwz(ZH, SFH, time=time)

A1 = mwa(age, d1['SFH'][:])
A2 = mwa(age, d2['SFH'][:])
A3 = mwa(age, SFH)

k = binmerger(X1, 10**np.linspace(np.log10(min(X1)), np.log10(max(X1)), 11))
kd = np.digitize(X1, k)
ck, cv, cw = count(kd)
a0 = np.asarray([10**np.median(np.log10(X1[o])) for o in cw])
a1 = np.asarray([10**np.median(np.log10(M1[o])) for o in cw])
a1e = (a1-np.asarray([10**np.percentile(np.log10(M1[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(M1[o]), 85) for o in cw])-a1)
a2 = np.asarray([10**np.median(np.log10(M2[o])) for o in cw])
a2e = (a2-np.asarray([10**np.percentile(np.log10(M2[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(M2[o]), 85) for o in cw])-a2)
k = binmerger(X3, 10**np.linspace(np.log10(min(X3)), np.log10(max(X3)), 11))
kd = np.digitize(X3, k)
ck, cv, cw = count(kd)
b0 = np.asarray([10**np.median(np.log10(X3[o])) for o in cw])
b2 = np.asarray([10**np.median(np.log10(M3[o])) for o in cw])
b2e = (b2-np.asarray([10**np.percentile(np.log10(M3[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(M3[o]), 85) for o in cw])-b2)

plt.figure(figsize=(10, 6))
P1=plt.scatter(X1, M1, s=1, color='red')
P2=plt.scatter(X2, M2, s=1, color='blue')
P3=plt.scatter(X3, M3, s=1, color='blue')
P4=plt.scatter(a0, a1, s=40, color='darkred', marker='D')
plt.errorbar(a0, a1, a1e, ecolor='darkred', fmt='none')
P5=plt.scatter(a0, a2, s=40, color='darkblue', marker='D')
plt.errorbar(a0, a2, a2e, ecolor='darkblue', fmt='none')
P6=plt.scatter(b0, b2, s=40, color='limegreen', marker='D')
plt.errorbar(b0, b2, b2e, ecolor='limegreen', fmt='none')
plt.yscale('log')
plt.xscale('log')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'Halo Mass ($M_{\odot}$)', fontsize=18)
plt.ylabel(r'Stellar Mass ($M_{\odot}$)', fontsize=18)
plt.ylim(1e9, 2e13)
plt.xlim(2e10, 3e15)
plt.legend([(P1, P4), (P2, P5), (P3, P6)], ['Original Data', 'Baryonic Prediction', 'Dark Prediction'], handler_map={P1:leghand(numpoints=2), P2:leghand(numpoints=2), P3:leghand(numpoints=2)}, loc=2, frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig('shmr-central-dark.pdf', bbox_inches = 'tight')
plt.close()

k = binmerger(M1, 10**np.linspace(np.log10(min(M1)), np.log10(max(M1)), 11))
kd = np.digitize(M1, k)
ck, cv, cw = count(kd)
b0 = np.asarray([10**np.median(np.log10(M1[o])) for o in cw])
b1 = np.asarray([10**np.median(np.log10(Z1[o])) for o in cw])
b1e = (b1-np.asarray([10**np.percentile(np.log10(Z1[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(Z1[o]), 85) for o in cw])-b1)
k = binmerger(M2, 10**np.linspace(np.log10(min(M2)), np.log10(max(M2)), 11))
kd = np.digitize(M2, k)
ck, cv, cw = count(kd)
c0 = np.asarray([10**np.median(np.log10(M2[o])) for o in cw])
c1 = np.asarray([10**np.median(np.log10(Z2[o])) for o in cw])
c1e = (c1-np.asarray([10**np.percentile(np.log10(Z2[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(Z2[o]), 85) for o in cw])-c1)
k = binmerger(M3, 10**np.linspace(np.log10(min(M3)), np.log10(max(M3)), 11))
kd = np.digitize(M3, k)
ck, cv, cw = count(kd)
d0 = np.asarray([10**np.median(np.log10(M3[o])) for o in cw])
d1 = np.asarray([10**np.median(np.log10(Z3[o])) for o in cw])
d1e = (d1-np.asarray([10**np.percentile(np.log10(Z3[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(Z3[o]), 85) for o in cw])-d1)

plt.figure(figsize=(10, 6))
P1=plt.scatter(M1, Z1, s=1, color='red', zorder=1)
P2=plt.scatter(M2, Z2, s=1, color='darkmagenta', zorder=3)
P3=plt.scatter(M3, Z3, s=1, color='teal', zorder=2)
P4=plt.scatter(b0, b1, s=40, color='darkred', marker='D', zorder=4)
plt.errorbar(b0, b1, b1e, ecolor='darkred', fmt='none', zorder=4)
P5=plt.scatter(c0, c1, s=40, color='darkviolet', marker='D', zorder=6)
plt.errorbar(c0, c1, c1e, ecolor='darkviolet', fmt='none', zorder=6)
P6=plt.scatter(d0, d1, s=40, color='deepskyblue', marker='D', zorder=5)
plt.errorbar(d0, d1, d1e, ecolor='deepskyblue', fmt='none', zorder=5)
plt.yscale('log')
plt.xscale('log')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'Stellar Mass ($M_{\odot}$)', fontsize=18)
plt.ylabel(r'Mass-Weighted Metallicity ($Z_{\odot}$)', fontsize=18)
plt.ylim(0.1, 3)
plt.xlim(1e9, 1e13)
plt.legend([(P1, P4), (P2, P5), (P3, P6)], ['Original Data', 'Baryonic Prediction', 'Dark Prediction'], handler_map={P1:leghand(numpoints=2), P2:leghand(numpoints=2), P3:leghand(numpoints=2)}, loc=2, frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig('mzr-central-dark.pdf', bbox_inches = 'tight')
plt.close()

digbins = np.percentile(M1, np.concatenate((np.linspace(0,90,10)[:-1], np.linspace(90, 100, 6))))
digbins[-1] *= 1.01
dig = np.digitize(M1, bins=digbins)
dk, dv, dw = count(dig)

ym = np.asarray([np.mean(M1[i]) for i in dw])
mwai = np.asarray([np.median(A1[i]) for i in dw])
mwaj = np.asarray([np.median(A2[j]) for j in dw])
mwais = np.asarray([iqr(A1[i]) for i in dw])
mwajs = np.asarray([iqr(A2[j]) for j in dw])

digbins = np.percentile(M3, np.concatenate((np.linspace(0,90,10)[:-1], np.linspace(90, 100, 6))))
digbins[-1] *= 1.01
dig = np.digitize(M3, bins=digbins)
dk, dv, dw = count(dig)

Ym = np.asarray([np.mean(M3[i]) for i in dw])
Mwai = np.asarray([np.median(A3[i]) for i in dw])
Mwais = np.asarray([iqr(A3[i]) for i in dw])

plt.scatter(ym, mwai, color='red', marker='D', label='Original Data', zorder=2)
plt.errorbar(ym, mwai, yerr=mwais, fmt='none', ecolor='red', zorder=1)
plt.scatter(ym, mwaj, color='blue', marker='D', label='Baryonic Prediction', zorder=2)
plt.errorbar(ym, mwaj, yerr=mwajs, fmt='none', ecolor='blue', zorder=1)
plt.scatter(Ym, Mwai, color='green', marker='D', label='Dark Prediction', zorder=2)
plt.errorbar(Ym, Mwai, yerr=Mwajs, fmt='none', ecolor='green', zorder=1)
plt.xscale('log')
plt.xlim(1e9, 2e12)
plt.tick_params(axis='both', labelsize=12)
plt.xlabel(r'Stellar Mass ($M_{\odot}$)', fontsize=16)
plt.ylabel('Stellar Mass Weighted Age (Gyr)', fontsize=16)
ylim=plt.gca().get_ylim()
plt.ylim(ylim)
plt.legend(loc=4, frameon=False)
plt.tight_layout()
plt.savefig('mwa-central-dark.pdf', bbox_inches = 'tight')
plt.close()
