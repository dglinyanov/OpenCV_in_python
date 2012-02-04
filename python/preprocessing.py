from cv import *
from opencv.highgui import *

import numpy as np

from util import *

def findX(image_source):
	min_found = False
	maxVal = image_source.shape[0] * 255
	#For each col sum, if sum < width*255 then we find the min 
	#then continue to end to search the max, if sum < width*255 then is new max
	for i in xrange(image_source.shape[0]):
		data = GetCol(fromarray(image_source), i)
		val = Sum(data)[0]		
		if val < maxVal:
			_max = i
			if not min_found:
				_min = i
				min_found = True
	return _min, _max

def findY(image_source):	
	min_found = False
	maxVal = image_source.shape[0] * 255
	#For each col sum, if sum < width*255 then we find the min 
	#then continue to end to search the max, if sum< width*255 then is new max
	for i in xrange(image_source.shape[0]):
		data = GetRow(fromarray(image_source), i)
		val = Sum(data)[0]
		if val < maxVal:
			_max = i
			if not min_found:
				_min = i
				min_found = True
	return _min, _max

def findRect(image_source):	
	x_min, x_max = findX(image_source)
	y_min, y_max = findY(image_source)
	rect = (x_min, y_min, x_max - x_min, y_max - y_min)	
	return rect

def preprocessing(image_source, new_width, new_height):
	#Find bounding box
	
	bb = findRect(image_source)
	#Get bounding box data and no with aspect ratio, the x and y can be corrupted
	data = GetSubRect(fromarray(image_source), (bb[0], bb[1], bb[2], bb[3]))
	#Classify sends data with 40 byte length rows
	#Create image with this data with width and height with aspect ratio 1 
	#then we get highest size betwen width and height of our bounding box
	size = bb[2] if bb[2] > bb[3] else bb[3]
	result = CreateImage(( size, size ), IPL_DEPTH_8U, 1)
	Set(result, 255)
	#Copy the data in center of image
	x = int( (size - bb[2]) / 2.0 - 0.5)
	y = int( (size - bb[3]) / 2.0 - 0.5)
	dataA = GetSubRect(result, (x, y, bb[2], bb[3]))
	Copy(data, dataA)
	#Scale result
	scaledResult = CreateImage((new_width, new_height ), IPL_DEPTH_8U, 1)
	Set(scaledResult, 0)	
	# Try CV_INTER_NN, CV_INTER_LINEAR, CV_INTER_AREA, CV_INTER_CUBIC
	Resize(result, scaledResult, CV_INTER_NN)
	
	#Return processed data
	return cv2array(scaledResult)