Team - 33
Project -10


Yash Goyal (201502181)
Mayank Garg (201530097)
Kumar Abhishek (201502172)
Dhruval Jain (201530109)

Classification of Book genres by Cover and Title
1. scrapping.py -> Dataset Generation from Openlibrary.org using Selenium and BeautifulSoup library.
2. makedataset.py -> Extracted book cover, title and genre.
3. resizeimage.py -> Resized all images to 224 x 224 x 3.
4. preprocess.py -> Compute Ratio b/w no. of thresholded pixels over the total pixel count. Discard all those images  where this ratio is less than 0.2.
5. Classifications:
In all the codes below we have extracted the feature vector of images using AlexNet and feature vector of titles using Word2Vec.
	5.1 svm(c=10).py -> Used SVM as a classifier.
	5.2 softmax.py -> Used Softmax as a classifier.
	5.3 adaboost(decision tree).py -> Used Decision tree with adaboost as a classifier.
6.recommendation.py -> Tried to recommend the 5 similar book titles depending the book title input by the user , but it ended up giving the same 5 results. So, it did not work for us, this maybe considered as future work.
