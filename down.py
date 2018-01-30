import urllib
import os
import cv2


#genre_list = ['science_fiction', 'fantasy', 'romance' , 'medicine' , 'business']

file = open("textbooksdata2.csv","r") 
l = file.readlines()
file.close()
f = open("/home/yash/processed_textbooksdata.csv","w")
base_dir = "/home/yash"
count = 0
for line in l:
	#c=c+1
	ls = line.split("||")
	genre = ls[2].strip()
	print genre
	link = "http://"+ls[1].strip("//") 
	img = link.split("/")
	img_name = img[-1]
	if "avatar_book" not in link:
	#print link
		img_loc =  base_dir+"/coverimages/"+genre+"/"+img_name
		print img_loc
		urllib.urlretrieve(link, img_loc)
	
		im = cv2.imread(img_loc,0)
		median = cv2.medianBlur(im,5)
		th2 = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
		rows,cols = th2.shape
		white = 0
		for i in range(rows):
			for j in range(cols):
				if th2[i,j] == 0:
					white+=1
		if float(white)/(rows*cols) <= 0.2:
			os.remove(img_loc)
			count +=1
		else:
			f.write(line)
f.close()

print count





