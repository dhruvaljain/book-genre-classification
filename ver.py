#import numpy as np
#import cv2
#import pandas as pd
#import sys
import os
#import urllib3

# count= {}
genre_list = ['business','fantasy','textbooks','romance','science_fiction']
# for i in genre_list:
# 	count[i]=0
file = open("/home/yash/processed_data.csv","r") 
l = file.readlines()
file.close()
#f = open("/home/dhruval/Desktop/processed_data.csv","w")
base_dir = "/home/yash/coverimages"

lis = []
for gen in genre_list:
	lis = lis + (os.listdir(base_dir+"/"+gen))


for line in l:
	ls = line.split("||")
	genre = ls[2].strip("\n")
	link = "http://"+ls[1].strip("//") 
	img = link.split("/")
	img_name = img[-1]
	#print genre
	#img_loc = base_dir+"/"+genre+"/"+img_name
	if img_name not in lis:
		print "hey"
		print img_name , genre 

#print lis
# 	median = cv2.medianBlur(im,5)
# 	th2 = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
# 	rows,cols = th2.shape
# 	white = 0
# 	for i in range(rows):
# 		for j in range(cols):
# 			if th2[i,j] == 0:
# 				white+=1
# 	if float(white)/(rows*cols) <= 0.2:
# 		os.remove(base_dir+"/"+genre+"/"+img_name)
# 		count[genre] +=1
# 	else:
# 		f.write(line)

# f.close()