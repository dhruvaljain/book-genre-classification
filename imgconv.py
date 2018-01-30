import os

from PIL import Image

from resizeimage import resizeimage

genre_list = ['business','fantasy','textbooks','romance','science_fiction']

base_dir = "/home/yash/coverimages"

for gen in genre_list:
	os.chdir(base_dir+"/"+gen)
	path = os.getcwd()
	lis = os.listdir(path)
	for file in lis:
		with open(file, 'r+b') as f:
			with Image.open(f) as image:
				w,h = image.size
				if  w == 224 and h == 224:
					print file
				# cover = resizeimage.resize_cover(image, [224, 224],validate = False)
				# cover.save(file, image.format)