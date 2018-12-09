from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation, Concatenate, Flatten, Lambda, Dot, Add, multiply, Conv2DTranspose
from keras.models import Model
from keras import layers
from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np
import cv2
import sys
import os
import multiprocessing

left_img = cv2.imread(sys.argv[1], 0).astype('float32') / 255
right_img = cv2.imread(sys.argv[2], 0).astype('float32') / 255
max_disp = int(sys.argv[3])

left_img = (left_img - left_img.mean()) / left_img.std()
right_img = (right_img - right_img.mean()) / right_img.std()

height, width = left_img.shape
channels = 1
kernel_size = 3
patchSize = 44
num_scale = 3

left_img = np.reshape(left_img, (height, width, channels))
right_img = np.reshape(right_img, (height, width, channels))

#height -= height % 4
#width -= width % 4
#left_img = left_img[0:height, 0:width]
#right_img = right_img[0:height, 0:width]

def run_gpu(left_img, right_img, height, width, channels):
	in_patch = Input(shape=(2, height, width, channels))

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

	model = Model(inputs = in_patch, outputs = [left, right])

	def categorical_hinge(y_true, y_pred):
		pos = K.sum(y_true * y_pred, axis=-1)
		neg = K.max((1. - y_true) * y_pred, axis=-1)
		return K.maximum(0., neg - pos + 0.2)

	model.compile(optimizer = 'adam', loss = categorical_hinge)

	model.load_weights('model/param')

	ret = model.predict(np.array([[left_img, right_img]]), verbose = 1)

	ret[0].tofile('tmp/left.bin')
	ret[1].tofile('tmp/right.bin')

if __name__ == '__main__':
	p = multiprocessing.Process(target = run_gpu, args = (left_img, right_img, height, width, channels))
	p.start()
	p.join()

	os.system("CreateCostVolumeDirect tmp\\left.bin tmp\\right.bin result\\left.png result\\right.png result\\disp.png " 
		+ str(width) + " " + str(height) + " 64 " + str(max_disp) + " " + str(patchSize / 2) + " 3 4 100 " + sys.argv[1] + " " + sys.argv[2] + " "
		+ str(sys.argv[4] if len(sys.argv) >= 5 else "") + " " + str(sys.argv[5] if len(sys.argv) >= 6 else ""))
