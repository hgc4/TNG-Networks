from basemodule import *
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import QuantileTransformer

def ReLU(x, maxval=1):
	return keras.backend.relu(x, max_value=maxval)

def offsets(data):
	Offsets = []
	for o in range(np.shape(data)[-1]):
		dat = data[...,o].ravel()
		udat = np.unique(dat)
		diff = np.diff(udat)
		Offsets.append([abs(diff[1] - diff[0]), abs(diff[-1] - diff[-2])])
	return np.asarray(Offsets)

def GQTvecnorm(data):
	'''
	Gaussian Quantile Transformation of 3D (temporal) dataset, with vector normalisation.
	'''
	GQT = QuantileTransformer(output_distribution='normal')
	datashape = np.shape(data)
	sample_ax, timestep_ax, quantity_ax = datashape
	Data = data.reshape(sample_ax*timestep_ax, quantity_ax)
	qData = GQT.fit_transform(Data)
	Offsets = offsets(qData)
	for o in range(len(Offsets)):
		wmin = np.where(qData[...,o]==np.min(qData[...,o]))
		wmax = np.where(qData[...,o]==np.max(qData[...,o]))
		qData[wmin] += Offsets[o,0]
		qData[wmax] -= Offsets[o,1]
	qData = qData.reshape(*datashape)
	return qData, GQT, Offsets

def GQTscalnorm(data, add_dim=False):
	'''
	Gaussian Quantile Transformation of 2D (non-temporal) or 3D (temporal) dataset, with scalar normalisation.
	'''
	GQT = QuantileTransformer(output_distribution='normal')
	datashape = np.shape(data)
	if len(datashape) > 2:
		sample_ax, timestep_ax, quantity_ax = datashape
		Data = data.reshape(sample_ax, quantity_ax*timestep_ax)
		qData = GQT.fit_transform(Data)
		Offsets = offsets(qData)
		for o in range(len(Offsets)):
			wmin = np.where(qData[...,o]==np.min(qData[...,o]))
			wmax = np.where(qData[...,o]==np.max(qData[...,o]))
			qData[wmin] += Offsets[o,0]
			qData[wmax] -= Offsets[o,1]
		qData = qData.reshape(*datashape)
	else:
		qData = GQT.fit_transform(data)
		Offsets = offsets(qData)
		for o in range(len(Offsets)):
			wmin = np.where(qData[...,o]==np.min(qData[...,o]))
			wmax = np.where(qData[...,o]==np.max(qData[...,o]))
			qData[wmin] += Offsets[o,0]
			qData[wmax] -= Offsets[o,1]
		if add_dim:
			qData = qData.reshape(*datashape, 1)
	return qData, GQT, Offsets

def GQTvecinv(qData, GQT, Offsets):
	'''
	Inverse Gaussian Quantile Transformation of 3D (temporal) dataset and GQT object, with vector normalisation.
	'''
	datashape = np.shape(qData)
	sample_ax, timestep_ax, quantity_ax = datashape
	qData = qData.reshape(sample_ax*timestep_ax, quantity_ax)
	for o in range(len(Offsets)):
		wmin = np.where(qData[...,o]==np.min(qData[...,o]))
		wmax = np.where(qData[...,o]==np.max(qData[...,o]))
		qData[wmin] -= Offsets[o,0]
		qData[wmax] += Offsets[o,1]
	data = GQT.inverse_transform(qData)
	return data.reshape(*datashape)

def GQTscalinv(qData, GQT, Offsets, add_dim=False):
	'''
	Inverse Gaussian Quantile Transformation of 2D (non-temporal) or 3D (temporal) dataset and GQT object, with scalar normalisation.
	'''
	datashape = np.shape(qData)
	for o in range(len(Offsets)):
		wmin = np.where(qData[...,o]==np.min(qData[...,o]))
		wmax = np.where(qData[...,o]==np.max(qData[...,o]))
		qData[wmin] -= Offsets[o,0]
		qData[wmax] += Offsets[o,1]
	if len(datashape) > 2:
		sample_ax, timestep_ax, quantity_ax = datashape
		qData = qData.reshape(sample_ax, quantity_ax*timestep_ax)
		data = GQT.inverse_transform(qData)
		return data.reshape(*datashape)
	else:
		data = GQT.inverse_transform(qData)
		if add_dim:
			qData = qData.reshape(*datashape, 1)
		return data

def lr_exp(E, lr0=0.001, k=0.1):
	'''
	Variable exponential learning rate, for recursive use in network training.
	'''
	if E > 0:
		lr0 *= np.exp(-k)
	return lr0

def lr_expcap(E, lr0=0.001, k=0.1, cap=6e-5):
	'''
	Capped exponential learning rate. Similar to lr exp, but with a minimum allowed value of the learning rate.
	'''
	if E > 0:
		lr0 *= np.exp(-k)
	return max([lr0, cap])

def lr_step(E, lr0=0.001, dr=0.5, Ed=10):
	'''
	Step-based variable learning rate. Learning rate is multiplied by a constant r at equally spaced intervals, E_d epochs apart.
	'''
	if E % Ed == 0 and E > 0:
		lr0 *= dr
	return lr0

def lr_step_r(E, lr0=0.001, dr=0.5, rp=0.25, Es=10):
	'''
	Random step function. For each epoch after a specified number, there is a probability that the learning rate is randomly multiplied by the step factor.
	'''
	rn = np.random.uniform()
	if rn <= rp and E > Es:
		lr0 *= dr
	return lr0

EPOCHS = int(1 + np.log(1000)/0.1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
scheduler = keras.callbacks.LearningRateScheduler(lr_exp, verbose=0)

def CentralNetwork():
	'''
	Compiles the network designed for use with central galaxies.
	'''
	input1 = keras.Input(shape=(33,8))
	input2 = keras.Input(shape=(9,))
	
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(input1)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu)(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	model1 = keras.Model(inputs=input1, outputs=dense1)
	
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(input2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	#drop2 = keras.layers.Dropout(rate=0.05)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	model2 = keras.Model(inputs=input2, outputs=dense2)
	
	inp = [model1.input, model2.input]
	conc = keras.layers.concatenate([model1.output, model2.output])
	dense = keras.layers.Dense(42, activation=tf.nn.elu)(conc)
	dense = keras.layers.Dense(42, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(42, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(42, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=keras.activations.linear)(dense)
	dense = keras.layers.Dense(69)(dense)
	model = keras.Model(inputs=inp, outputs=dense)
	
	optimizer = keras.optimizers.RMSprop(0.0007)
	
	model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
	
	return model

def SatelliteNetwork():
	'''
	Compiles the network designed for use with satellite galaxies.
	'''
	input1 = keras.Input(shape=(33,7))
	input2 = keras.Input(shape=(11,))
	
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(input1)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu)(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	model1 = keras.Model(inputs=input1, outputs=dense1)
	
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(input2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	#drop2 = keras.layers.Dropout(rate=0.05)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	model2 = keras.Model(inputs=input2, outputs=dense2)
	
	inp = [model1.input, model2.input]
	conc = keras.layers.concatenate([model1.output, model2.output])
	dense = keras.layers.Dense(44, activation=tf.nn.elu)(conc)
	dense = keras.layers.Dense(44, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(44, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(44, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=keras.activations.linear)(dense)
	dense = keras.layers.Dense(69)(dense)
	model = keras.Model(inputs=inp, outputs=dense)
	
	optimizer = keras.optimizers.RMSprop(0.001)
	
	model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
	
	return model
