from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation, Concatenate, Flatten, Lambda, Dot, Add, multiply, Conv2DTranspose
from keras.models import Model
from keras import layers
from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np
import cv2
import sys

batchSize = 32
patchSize = 44
channels = 1
kernel_size = 3
num_scale = 3
useDataList = ['coel', 'MB2001', 'MB2003', 'MB2005', 'MB2006_1', 'MB2006_2', 'MB2014']
shuffleDataset = True

in_patch = Input(shape=(2, patchSize, patchSize, channels))

def take_left(x):
	return x[:, 0]
def take_right(x):
	return x[:, 1]

left = Lambda(take_left)(in_patch)
right = Lambda(take_right)(in_patch)

patchlist = [[left, right]]

for i in range(1, num_scale):
	left_sm = MaxPooling2D(2 ** i, padding = 'same')(left)
	right_sm = MaxPooling2D(2 ** i, padding = 'same')(right)

	patchlist.append([left_sm, right_sm])

for patch in patchlist:
	left = patch[0]
	right = patch[1]

	for j in range(4):
		conv = Conv2D(64, (kernel_size, kernel_size), padding = 'same')
		act = Activation('relu')

		left = conv(left)
		left = BatchNormalization()(left)
		left = act(left)

		right = conv(right)
		right = BatchNormalization()(right)
		right = act(right)

	patch[0] = left
	patch[1] = right

left = patchlist[0][0]
right = patchlist[0][1]

for i in range(1, num_scale):
	left_sm = patchlist[i][0]
	right_sm = patchlist[i][1]

	left_sm = UpSampling2D(2 ** i)(left_sm)
	right_sm = UpSampling2D(2 ** i)(right_sm)

	left = Concatenate()([left, left_sm])
	right = Concatenate()([right, right_sm])

for j in range(2):
	conv = Conv2D(64, (kernel_size, kernel_size), padding = 'same')
	act = Activation('relu')

	left = conv(left)
	left = BatchNormalization()(left)
	left = act(left)

	right = conv(right)
	right = BatchNormalization()(right)
	right = act(right)

conv = Conv2D(64, (kernel_size, kernel_size), padding = 'same')

left = conv(left)
right = conv(right)

left_out = Flatten()(left)
right_out = Flatten()(right)
out = Dot(-1, True)([left_out, right_out])

model = Model(inputs = in_patch, outputs = out)

def categorical_hinge(y_true, y_pred):
	pos = K.sum(y_true * y_pred, axis=-1)
	neg = K.max((1. - y_true) * y_pred, axis=-1)
	return K.maximum(0., neg - pos + 0.2)

model.compile(optimizer = 'adam', loss = categorical_hinge)
#model.load_weights('model/param0002')

def min_max(x, axis=None):
	min = x.min(axis=axis, keepdims=True)
	max = x.max(axis=axis, keepdims=True)
	result = (x-min)/(max-min)
	return result

file = open('loss.csv', mode = 'w')
for x in sorted(useDataList):
	file.write(x + ',')
file.write('\n')
file.close()

for it in range(0, 10):
	if shuffleDataset:
		np.random.shuffle(useDataList)

	loss_dict = {}
	for dataset in useDataList:
		print(dataset)

		x_train = np.reshape(np.fromfile("../../Dataset/Stereo/Aug/" + str(patchSize) + "/positive/" + dataset, np.float32), (-1, 2, patchSize, patchSize, channels))
		x_train = np.concatenate([x_train, np.reshape(np.fromfile("../../Dataset/Stereo/Aug/" + str(patchSize) + "/negative/" + dataset, np.float32), (-1, 2, patchSize, patchSize, channels))])
		y_train = np.concatenate([np.ones(int(len(x_train) / 2)), np.zeros(int(len(x_train) / 2))])

		print(x_train.shape)
		print(y_train.shape)

#		for patches in x_train:
#			cv2.imshow("left", min_max(patches[0]))
#			cv2.imshow("right", min_max(patches[1]))
#			cv2.waitKey()

		his = model.fit(x_train, y_train,
				epochs = 1,
				batch_size = batchSize,
				shuffle = True,
#				validation_data = (x_test, y_test),
				callbacks=[TensorBoard(log_dir='log')])

		loss_dict[dataset] = his.history['loss'][0]

		del x_train
		del y_train

	file = open('loss.csv', mode = 'a')
	for k, v in sorted(loss_dict.items()):
		file.write(str(v) + ',')
	file.write('\n')
	file.close()

	model.save_weights('model/param%04d' % (it))
