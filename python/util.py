from cv import *
from cv2 import *
import numpy as np

def cv2array(im): 
  	depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8', 
        cv.IPL_DEPTH_8S: 'int8', 
        cv.IPL_DEPTH_16U: 'uint16', 
        cv.IPL_DEPTH_16S: 'int16', 
        cv.IPL_DEPTH_32S: 'int32', 
        cv.IPL_DEPTH_32F: 'float32', 
        cv.IPL_DEPTH_64F: 'float64', 
    } 

  	arrdtype=im.depth 
  	a = np.fromstring( 
         im.tostring(), 
         dtype=depth2dtype[im.depth], 
         count=im.width*im.height*im.nChannels) 
  	a.shape = (im.height,im.width,im.nChannels) 
  	return a 

def row2array(row):
  	arrdtype='uint8'
  	a = np.fromstring( 
         row.tostring(), 
         dtype='uint8', 
         count=row.width*row.height) 
  	a.shape = (row.height,row.width,1) 
  	return a 


def debug_print_mat(src):
	f = open('output', 'a')
	
	line = 0
	
	for i in xrange(src.height):
		for j in xrange(src.width):			
			if line == 40:
				f.write('\n')
				line = 0
			f.write(str(int(src[i, j])))
			f.write(' ')
			line += 1

	f.write('\n\n')
	f.close()
	sys.exit(0)


def debug_print_np(src):
	f = open('output', 'w')
	f.truncate(0)
	line = 0
	for c in src:
		for x in c:			
			if line == 40:
				f.write('\n')
				line = 0
			f.write(str(int(x)))
			f.write(' ')
			line += 1
	f.close()
	#sys.exit(0)