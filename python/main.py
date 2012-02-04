#!/usr/bin/python
from cv import *
import cv
from cv2 import *
from basicOCR import basicOCR
from util import cv2array

class Main():
	def __init__(self):
		self.drawing = 0
		self.r = 10
		self.red = self.green = self.blue = 0
		self.last_x = self.last_y = 0 
		#Create image
		self.imagen = CreateImage((128,128), IPL_DEPTH_8U, 1)
		#Set data of image to white
		Set(self.imagen, CV_RGB(255,255,255))
		#Image we show user with cursor and other artefacts we need
		self.screenBuffer = CloneImage(self.imagen)
		#Create window
		NamedWindow( "Demo", 0 )
		ResizeWindow("Demo", 128,128)
		SetMouseCallback("Demo", self.on_mouse, 0)
		####################
		#My OCR
		####################
		self.ocr = basicOCR()

	def draw(self, x, y):
		#Draw a circle where is the mouse
		Circle(self.imagen, (x, y), self.r, CV_RGB(self.red, self.green, self.blue), -1, 4, 0)
		#Get clean copy of image
		self.screenBuffer = CloneImage(self.imagen)
		ShowImage( "Demo", self.screenBuffer )

	def drawCursor(self, x, y):
		#Get clean copy of image
		self.screenBuffer = CloneImage(self.imagen)
		#Draw a circle where is the mouse
		Circle(self.screenBuffer, (x,y), self.r, CV_RGB(0,0,0), 1, 4, 0);
	
	def on_mouse(self, event, x, y, flags, param ):
		self.last_x = x
		self.last_y = y
		self.drawCursor(x,y)
		#Select mouse Event
		if event == CV_EVENT_LBUTTONDOWN:
			self.drawing = 1
			self.draw(x, y)
		elif event == CV_EVENT_LBUTTONUP:
			#drawing=!drawing;
			self.drawing = 0
		elif event == CV_EVENT_MOUSEMOVE  and  (flags & CV_EVENT_FLAG_LBUTTON):
			if self.drawing:
				self.draw(x, y)

	def run(self):
		while 1:
			ShowImage( "Demo", self.screenBuffer );
			c = waitKey(10);
			if c == 27:
				break
			if c < -0 or c > 255:
				continue
			if chr(c) == '+':
				self.r += 1
				self.drawCursor(self.last_x, self.last_y)
			if chr(c) == '-' and self.r > 1:
				self.r -= 1
				self.drawCursor(self.last_x, self.last_y)
			if chr(c) == 'r':
				Set(self.imagen, RealScalar(255))
				self.drawCursor(self.last_x, self.last_y)
			if chr(c) == 's':
				SaveImage("out.png", self.imagen)
			if chr(c) == 'c':
				self.ocr.classify(cv2array(self.imagen), 1)				

		destroyWindow("Demo")
		return 0

main = Main()
main.run()
