from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import initializers
from keras import optimizers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree  import DecisionTreeClassifier
import os
import cv2
import sklearn
import numpy as np
from sklearn import svm
import gensim 
import nltk 
from nltk.corpus import stopwords

coding= 'utf-8' 

num_classes = 5

batch_size = 128

epochs = 200

file = open("processed_data.csv","r") 
l = file.readlines()
file.close()

base_dir = "/home/yash.goyal/coverimages"

genre_list = ['business','fantasy','textbooks','romance','science_fiction']
lab = {}
c=0
for i in genre_list:
	lab[i] = c
	c+=1

train = {}
traindir = "/home/yash.goyal/dataset/train/"
for gen in genre_list:
	train[gen] = os.listdir(traindir+gen)

test = {}
testdir = "/home/yash.goyal/dataset/test/"
for gen in genre_list:
	test[gen] = os.listdir(testdir+gen)

x_train = []
y_train = []
vec_train =[]
vec_test = []

x_test = []
y_test = []

for line in l:
	ls = line.split("||")
	title = ls[0]
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
		vec_train.append(title)
		x_train.append(im)
		y_train.append(lab[genre])
	else:
		vec_test.append(title)
		x_test.append(im)
		y_test.append(lab[genre])

x_train= np.asarray(x_train)
x_test = np.asarray(x_test)

x_train = x_train.reshape(x_train.shape[0],  224, 224 , 3)

x_test = x_test.reshape(x_test.shape[0],  224, 224 , 3)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = (224, 224 ,3)

model = Sequential()

model.add(Conv2D(3, kernel_size=(11,11),strides=4,activation='relu',input_shape=input_shape,kernel_initializer=initializers.random_normal(mean=0.0,stddev=0.01),bias_initializer='zeros',padding='same'))

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

sgd = optimizers.SGD(lr=0.0001, decay=0.0005, momentum=0.9)



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1)

clf = K.function([model.layers[0].input, K.learning_phase()],[model.layers[13].output])

svmtrain_x = []

svmtest_x = []



for i in xrange(len(x_train)):
	svmtrain_x.append(clf([x_train[i].reshape(1,224,224,3),0])[0].flatten())

for i in xrange(len(x_test)):
	svmtest_x.append(clf([x_test[i].reshape(1,224,224,3),0])[0].flatten())

#print y_train[0:5]

svmtrain_x= np.asarray(svmtrain_x)
svmtest_x= np.asarray(svmtest_x)

ytest = []
ytrain = []

for i in xrange(len(y_train)):
	x = y_train[i].tolist().index(1)
	ytrain.append(x)

for i in xrange(len(y_test)):
        x = y_test[i].tolist().index(1)
        ytest.append(x)


ytrain = np.asarray(ytrain)
ytest = np.asarray(ytest)

#word2vec
f_train = []
f_test = []
mod = gensim.models.KeyedVectors.load_word2vec_format('/home/yash.goyal/GoogleNews-vectors-negative300.bin', binary=True)  
stop_words = set(stopwords.words('english'))
# for i in xrange(len(vec_train)):
# 	tokens = nltk.word_tokenize(vec_train[i].decode('utf-8'))
# 	fs = [w for w in tokens if not w in stop_words]
# 	fs = map(lambda x:x.lower(),fs)
# 	tmp = []
# 	for j in xrange(len(genre_list)):
# 		t =0
# 		no = 0
# 		g = genre_list[j]
# 		if g == "science_fiction":
# 			for l in fs:
# 				if l in mod.wv.vocab:
# 					val = (mod.similarity("science",l) + mod.similarity("fiction",l))/2.0
# 					t = t+val
# 					no = no +1
# 		else:
# 			for l in fs:
# 				if l in mod.wv.vocab:
# 					val = mod.similarity(g,l)
# 					t = t+val
# 					no = no + 1
# 		if no!=0:
# 			p=t/no
# 		else:
# 			p=0
# 			print tokens
# 		tmp.append(p)
# 	f_train.append(tmp)

for i in xrange(len(vec_train)):
	tokens = nltk.word_tokenize(vec_train[i].decode('utf-8'))
	fs = [w for w in tokens if not w in stop_words]
	fs = map(lambda x:x.lower(),fs)
	tmp = np.zeros((1,300))
	tmp1 = np.zeros((1,300))
	no = 0
	for j in fs:
		if j in mod.wv.vocab:
			x =np.array(mod[j])
			np.add(x,tmp,out=tmp)
			no = no+1
	if no!=0:
		np.divide(tmp,no,out=tmp1)
	f_train.append(tmp1)


for i in xrange(len(vec_test)):
	tokens = nltk.word_tokenize(vec_test[i].decode('utf-8'))
	fs = [w for w in tokens if not w in stop_words]
	fs = map(lambda x:x.lower(),fs)
	tmp = np.zeros((1,300))
	tmp1 = np.zeros((1,300))
	no = 0
	for j in fs:
		if j in mod.wv.vocab:
			x =np.array(mod[j])
			np.add(x,tmp,out=tmp)
			no = no+1
	if no!=0:
		np.divide(tmp,no,out=tmp1)
	f_test.append(tmp1)
f_train = np.asarray(f_train)
f_test = np.asarray(f_test)

print f_train.shape
print f_test.shape


# for i in xrange(len(vec_test)):
# 	tokens = nltk.word_tokenize(vec_test[i].decode('utf-8'))
# 	fs = [w for w in tokens if not w in stop_words]
# 	fs = map(lambda x:x.lower(),fs)
# 	tmp = []
# 	for j in xrange(len(genre_list)):
# 		t =0
# 		no = 0
# 		g = genre_list[j]
# 		if g == "science_fiction":
# 			for l in fs:
# 				if l in mod.wv.vocab:
# 					val = (mod.similarity("science",l) + mod.similarity("fiction",l))/2.0
# 					t = t+val
# 					no = no +1
# 		else:
# 			for l in fs:
# 				if l in mod.wv.vocab:
# 					val = mod.similarity(g,l)
# 					t = t+val
# 					no = no + 1
# 		if no!=0:
# 			p = t/no
# 		else:
# 			p = 0
# 			print tokens
# 		tmp.append(p)
# 	f_test.append(tmp)

final_train = []
final_test = []

for i in xrange(len(x_train)):
 	tv = []
 	for j in range(1,7997):
 		if j%2==0 and j<=7800:
 			ind = (((j%600)/2)-1) % 300
 			tv.append(f_train[i][0][ind])
 		else:
 			tv.append(svmtrain_x[i][(j-1)/2])
 	final_train.append(tv)


for i in xrange(len(x_test)):
 	tv = []
 	for j in range(1,7997):
 		if j%2==0 and j<=7800:
 			ind = (((j%600)/2)-1) % 300
 			tv.append(f_test[i][0][ind])
 		else:
 			tv.append(svmtest_x[i][(j-1)/2])
 	final_test.append(tv)
# for i in xrange(len(x_test)):
# 	tv = []
# 	for j in range(1,8192):
# 		if j%2==0:
# 			ind = (((j%10)/2)-1) % 5
# 			tv.append(f_test[i][ind])
# 		else:
# 			tv.append(svmtest_x[i][(j-1)/2])
# 	final_test.append(tv)

final_train = np.asarray(final_train)
final_test = np.asarray(final_test)


#print final_train.shape
#print final_test.shape

#print final_train[0]
bdt_real=sklearn.linear_model.LogisticRegression()

# bdt_real = AdaBoostClassifier(
    # # DecisionTreeClassifier(max_depth=2),
    # n_estimators=600,
    # learning_rate=1)



bdt_real.fit(final_train,ytrain)

trainacc =bdt_real.predict(final_train)
train_accuracy = accuracy_score(trainacc, ytrain)

print "Train accuracy:" , train_accuracy

yPredarr = bdt_real.predict(final_test)
test_accuracy = accuracy_score(yPredarr, ytest)

print "Test accuracy:" , test_accuracy

array= sklearn.metrics.confusion_matrix(ytest, yPredarr, labels=None, sample_weight=None)

print "Confusion matrix for Testing" 
print array


array1= sklearn.metrics.confusion_matrix(ytrain, trainacc, labels=None, sample_weight=None)
print "Confusion matrix for Training" 
print array1

