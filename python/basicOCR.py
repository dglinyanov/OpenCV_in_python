import sys
from cv import *
from cv2 import *
from opencv.highgui import *
import numpy as np
from util import *
from preprocessing import *

# private:
# 		char file_path[255];
# 		int train_samples;
# 		int classes;
# 		CvMat* trainData;
# 		CvMat* trainClasses;
# 		int size;
# 		static const int K=10;
# 		CvKNearest *knn;
# 		void getData();
# 		void train();

class basicOCR():
	def __init__(self):
		
		self.__K = 10
		self.__file_path = "../OCR/"
		self.__train_samples = 50;
		self.__classes= 10;
		self.__size = 40;

		self.__trainData = CreateMat(self.__train_samples * self.__classes, self.__size * self.__size, CV_32FC1);
		self.__trainClasses = CreateMat(self.__train_samples * self.__classes, 1, CV_32FC1);

		#Get data (get images and process it)
		self.__getData();
	
		#train	
		self.__train();
		#Test	
		self.test();
	
		print " ---------------------------------------------------------------\n"
		print "|\tClass\t|\tPrecision\t|\tAccuracy\t|\n"
		print " ---------------------------------------------------------------\n"

	def __getData(self):		
		for i in xrange(self.__classes):
			for j in xrange(self.__train_samples):				
				#Load file TODO: rewrite this piece of shit
				if j < 10:
					_file = self.__file_path + str(i) + "/" + str(i) + "0" + str(j) + ".pbm"
				else:
					_file = self.__file_path + str(i) + "/" + str(i) + str(j) + ".pbm"
				src_image = imread(_file, 0)

				#process file
				prs_image = preprocessing(src_image, self.__size, self.__size)
				
				#Set class label
				row = GetRow(self.__trainClasses, i * self.__train_samples + j)				
				Set(row, i)				
				
				#Set data
				row = GetRow(self.__trainData, i * self.__train_samples + j)				

				img = CreateImage((self.__size, self.__size), IPL_DEPTH_32F, 1 )
				#convert 8 bits image to 32 float image
				ConvertScale(fromarray(prs_image), img, scale=(1.0/255))

				data = GetSubRect(img, (0, 0, self.__size, self.__size))
				
				#convert data matrix sizexsize to vecor
				row1 = Reshape(data, 0, 1)
				Copy(row1, row)
				
	
	def __train(self):
		data = np.array(self.__trainData, dtype=np.float32)		
		classes = np.array(self.__trainClasses, dtype=np.int32)
		zero = np.array(0, dtype=np.int32)
		self.__knn = KNearest()
		self.__knn.train(data, classes, None, False, self.__K)

	def classify(self, image_source, show_result):		
		nearest = CreateMat(1, self.__K,CV_32FC1)
		#process file	

		prs_image = preprocessing(image_source, self.__size, self.__size)
		
		#Set data 
		img32 = CreateImage((self.__size, self.__size), IPL_DEPTH_32F, 1)
		ConvertScale(fromarray(prs_image), img32, scale=(1.0/255))		
		data = GetSubRect(img32, (0, 0, self.__size, self.__size))
		row1 = Reshape(data, 0, 1)
		row1np = np.array(row1)
		retval, result, nearest, dists = self.__knn.find_nearest(row1np, self.__K)		
		
		accuracy = 0
		for i in xrange(self.__K):
			if nearest[0][i] == result[0][0]:
	                    accuracy += 1
		pre = 100 * accuracy / float(self.__K)
		if show_result == 1:
			print "|\t" + str(result[0][0]) + "\t| \t" + str(pre) + "  \t| \t" + str(accuracy) + " of " + str(self.__K) + " \t|"
			print " ---------------------------------------------------------------"

		return result

	def test(self):
		error = 0
		testCount = 0
		for  i in xrange(self.__classes):
			for j in xrange(50, 50 + self.__train_samples):
				_file = self.__file_path + str(i) + "/" + str(i) + str(j) + ".pbm"
				src_image = imread(_file, 0)				

				#process file
				prs_image = preprocessing(src_image, self.__size, self.__size)
				prs_np = prs_image
				r = self.classify(prs_np, 0)				
				if int(r) != i:
					error += 1
				
				testCount += 1

		totalerror = 100 * error / float(testCount)
		print "System Error: " + str(totalerror)




