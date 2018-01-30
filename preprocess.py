import numpy as np
import cv2
import pandas as pd
import sys
import os
import urllib3

data = sys.argv[1] 
data = pd.read_csv(data,header=None,delimiter='|')
im = cv2.imread("book2.jpg",0)
title = data.iloc[:,0].values
links = data.iloc[:,1].values
genre = data.iloc[:,2].values
for i in range(len(links)):
	links[i] = links[i][2:]

# for i in range(len(links)):
# 	title[i] = '_'.join(title[i].split())
# 	fullfilename = "/home/dhruval/Desktop/bookcovers/"+str(genre[i])+"/"+str(title[i])
# 	os.system("wget "+links[i]+" -O "+ fullfilename)

applicable = 0
genre_list = ['business','fantasy','medicine','romance','science_fiction']
for genres in genre_list:
	cur_dir = "/home/yash/coverimages/"+str(genres)	
	for filename in os.listdir(cur_dir):
		#print filename
		im = cv2.imread(cur_dir + "/" + filename,0)
		# cv2.imshow('output',im)
		median = cv2.medianBlur(im,5)
		th2 = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
		rows,cols = th2.shape
		white = 0
		for i in range(rows):
			for j in range(cols):
				if th2[i,j] == 0:
					white+=1
		if float(white)/(rows*cols) > 0.2:
			applicable+=1
		else:
			print filename
		 	os.rename(cur_dir+"/"+filename ,  cur_dir+"/"+"$"+filename)
			
# # cv2.imshow('output',th2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
