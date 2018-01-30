import bs4 as bs

import time

import csv

from selenium import webdriver

from urllib2 import urlopen

da = {}

genre_list = ['science_fiction', 'fantasy', 'romance' , 'textbooks' , 'business']

#genre_list = ['science_fiction']
base_add = "https://openlibrary.org/subjects/"

p = 170

browser = webdriver.Firefox()



for gen in genre_list:

	adr = base_add + gen

	browser.get(adr)

	for i in range(p):
		browser.find_element_by_xpath("/html/body/div[3]/div[2]/div/div[1]/div[5]/div/div[1]/div/div[2]").click()
		time.sleep(12)

	content2 = browser.page_source

	soup2 = bs.BeautifulSoup(content2,"lxml")

	productDivs2 = soup2.findAll('div', attrs={'class' : 'SRPCover'})
	for div in productDivs2:
		name = div.find('img')['alt']
		image = div.find('img')['src']
		if name not in da:
			da[name] = (image, gen)

with open("finaldata.csv", "wb") as csv_file:
		for i in da.keys():
			wr = str(i.encode('utf-8') + "||"+ da[i][0] + "||"+ da[i][1])
			csv_file.write(wr)
			csv_file.write('\n')
