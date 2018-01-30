import shutil
import os
import random

source = '/home/yash/coverimages'
train = '/home/yash/dataset/train'
test = '/home/yash/dataset/test'

genre_list = ['business','fantasy','textbooks','romance','science_fiction']

for gen in genre_list:

	files = os.listdir(source+"/"+gen)
	#print len(files)

	for f in files:
		x = random.randint(1,121)
		print x
		if x <= 20:
			#print "test"
			shutil.copy(source+"/"+gen+"/"+f, test+"/"+gen+"/")
		else :
			#print "train"
			shutil.copy(source+"/"+gen+"/"+f, train + "/"+gen+"/")