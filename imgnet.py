from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import initializers
from keras import optimizers
import os
import cv2
import numpy as np

num_classes = 5

batch_size = 128

epochs = 400

file = open("processed_data.csv","r") 
l = file.readlines()
file.close()

base_dir = "/home/yash/coverimages"

genre_list = ['business','fantasy','textbooks','romance','science_fiction']
lab = {}
c=0
for i in genre_list:
	lab[i] = c
	c+=1

train = {}
traindir = "/home/yash/dataset/train/"
for gen in genre_list:
	train[gen] = os.listdir(traindir+gen)

test = {}
testdir = "/home/yash/dataset/test/"
for gen in genre_list:
	test[gen] = os.listdir(testdir+gen)

x_train = []
y_train = []

x_test = []
y_test = []

for line in l:
	#c=c+1
	ls = line.split("||")
	genre = ls[2].strip()
	#print genre
	link = "http://"+ls[1].strip("//") 
	img = link.split("/")
	img_name = img[-1]
	#if "avatar_book" not in link:
	#print link
	img_loc =  base_dir+"/"+genre+"/"+img_name

	im = cv2.imread(img_loc)
	if img_name in train[genre]:
		x_train.append(im)
		y_train.append(lab[genre])
	else:
		x_test.append(im)
		y_test.append(lab[genre])

x_train= np.asarray(x_train)
x_test = np.asarray(x_test)

x_train = x_train.reshape(x_train.shape[0],  224, 224 , 3)

x_test = x_test.reshape(x_test.shape[0],  224, 224 , 3)

print (x_train.shape)
print (x_test.shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print (y_train.shape)
print (y_test.shape)
#print (y_train)

input_shape = (224, 224 ,3)

model = Sequential()

model.add(Conv2D(3, kernel_size=(11,11),strides=4,activation='relu',input_shape=input_shape,kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.01),bias_initializer='zeros',padding='same'))

#model.add(Dense(num_classes))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

model.add(Conv2D(96, (5, 5), activation='relu',kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.01),bias_initializer='ones',padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

model.add(Conv2D(256, (3, 3), activation='relu',kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.01),bias_initializer='zeros',padding='same'))

model.add(Conv2D(384, (3, 3), activation='relu',kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.01),bias_initializer='ones',padding='same'))

model.add(Conv2D(384, (3, 3), activation='relu',kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.01),bias_initializer='ones',padding='same'))

model.add(MaxPooling2D(pool_size=(3, 3),strides=2))

model.add(Flatten())

model.add(Dense(4096, activation='relu',kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.01),bias_initializer='zeros'))

model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu',kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.01),bias_initializer='zeros'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

sgd = optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)