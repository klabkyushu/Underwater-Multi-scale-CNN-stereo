from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation, Concatenate, Flatten, Lambda, Dot, Conv2DTranspose
from keras.models import Model
from keras import layers
from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np
import cv2
import sys
import os

def makeModel_fcn(height, width, channels, kernel_size):
	in_patch = Input(shape=(2, height, width, channels))

	def take_left(x):
	    return x[:, 0]
	def take_right(x):
	    return x[:, 1]

	left = Lambda(take_left)(in_patch)
	right = Lambda(take_right)(in_patch)

	left_sm = Conv2D(channels, (kernel_size, kernel_size), padding = 'same', strides = 2)(left)
	right_sm = Conv2D(channels, (kernel_size, kernel_size), padding = 'same', strides = 2)(right)
	print(left_sm.shape)

	for j in range(2):
		left_sm = Conv2D(64, (kernel_size, kernel_size), padding = 'same')(left_sm)
		left_sm = BatchNormalization()(left_sm)
		left_sm = Activation('relu')(left_sm)

		right_sm = Conv2D(64, (kernel_size, kernel_size), padding = 'same')(right_sm)
		right_sm = BatchNormalization()(right_sm)
		right_sm = Activation('relu')(right_sm)

	left_sm = Conv2DTranspose(64, (kernel_size, kernel_size), padding = 'same', strides = 2)(left_sm)
	right_sm = Conv2DTranspose(64, (kernel_size, kernel_size), padding = 'same', strides = 2)(right_sm)

	for j in range(2):
		left = Conv2D(64, (kernel_size, kernel_size), padding = 'same')(left)
		left = BatchNormalization()(left)
		left = Activation('relu')(left)

		right = Conv2D(64, (kernel_size, kernel_size), padding = 'same')(right)
		right = BatchNormalization()(right)
		right = Activation('relu')(right)

	left = Concatenate()([left, left_sm])
	right = Concatenate()([right, right_sm])

	for j in range(2):
		left = Conv2D(64, (kernel_size, kernel_size), padding = 'same')(left)
		left = BatchNormalization()(left)
		left = Activation('relu')(left)

		right = Conv2D(64, (kernel_size, kernel_size), padding = 'same')(right)
		right = BatchNormalization()(right)
		right = Activation('relu')(right)

	left = Conv2D(64, (kernel_size, kernel_size), padding = 'same')(left)
	right = Conv2D(64, (kernel_size, kernel_size), padding = 'same')(right)

	model = Model(inputs = in_patch, outputs = [left, right])

	def categorical_hinge(y_true, y_pred):
		pos = K.sum(y_true * y_pred, axis=-1)
		neg = K.max((1. - y_true) * y_pred, axis=-1)
		return K.maximum(0., neg - pos + 0.2)

	model.compile(optimizer = 'adam', loss = categorical_hinge)

	return model
