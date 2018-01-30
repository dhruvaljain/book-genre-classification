
import cv2
import os
# data = sys.argv[1] 
# with open("finaldata.csv") as f:
# 	lis = [line.split() for line in f]
# data = pd.read_csv('finaldata.csv',header=None,sep="||")
# im = cv2.imread("book2.jpg",0)
# title = data.iloc[:,0].values
# links = data.iloc[:,1].values
# genre = data.iloc[:,2].values
# for i in range(len(links)):
# 	links[i] = links[i][2:]
count= {}
genre_list = ['business','fantasy','textbooks','romance','science_fiction']
for i in genre_list:
	count[i]=0
file = open("/home/yash.goyal/finaldata.csv","r") 
l = file.readlines()
file.close()
f = open("/home/yash.goyal/processed_data.csv","w")

base_dir = "/home/yash.goyal/coverimages"
for line in l:
	ls = line.split("||")
	genre = ls[2].strip("\n")
	link = "http://"+ls[1].strip("//") 
	img = link.split("/")
	img_name = img[-1]
	im = cv2.imread(base_dir+"/"+genre+"/"+img_name,0)
	median = cv2.medianBlur(im,5)
	th2 = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
	rows,cols = th2.shape
	white = 0
	for i in range(rows):
		for j in range(cols):
			if th2[i,j] == 0:
				white+=1
	if float(white)/(rows*cols) <= 0.2:
		os.remove(base_dir+"/"+genre+"/"+img_name)
		count[genre] +=1
	else:
		f.write(line)

f.close()


		
	#print img_name
	#urllib.urlretrieve(link, base_dir+"/coverimages/"+genre+"/"+img_name)	



# for i in range(len(links)):
# 	title[i] = '_'.join(title[i].split())
# 	fullfilename = "/home/dhruval/Desktop/bookcovers/"+str(genre[i])+"/"+str(title[i])
# 	os.system("wget "+links[i]+" -O "+ fullfilename)

# applicable = 0
# bekaar = 0
# genre_list = ['business','fantasy','history','romance','science_fiction']
# for genres in genre_list:
# 	cur_dir = "/home/dhruval/Desktop/bookcovers/"+str(genres)	
# 	for filename in os.listdir(cur_dir):
# 		#print filename
# 		im = cv2.imread(cur_dir + "/" + filename,0)
# 		# cv2.imshow('output',im)
# 		median = cv2.medianBlur(im,5)
# 		th2 = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
# 		rows,cols = th2.shape
# 		white = 0
# 		for i in range(rows):
# 			for j in range(cols):
# 				if th2[i,j] == 0:
# 					white+=1
# 		if float(white)/(rows*cols) > 0.2:
# 			applicable+=1
# 		else:
# 			print filename
# 			bekaar+=1
		 	# os.rename(cur_dir+"/"+filename ,  cur_dir+"/"+"$"+filename)
			
# # cv2.imshow('output',th2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print count
