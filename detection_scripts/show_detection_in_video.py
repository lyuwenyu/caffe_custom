import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io
import glob
import os




def show_detection_video(data_dir, det_file, result_name):



	frame = plt.imread(data_dir+'/000001.png')


	with open(det_file, 'r') as f:

		raw_data = f.readlines()
		raw_data = [l.strip().split(' ') for l in raw_data]

		
		det_result = {}

		for ll in raw_data:
			try:
				det_result[int(ll[0])].append(ll)
			except:
				det_result[int(ll[0])] = [ ll ]

		print len(det_result.keys())
		# print len(det_result['000001'])

		im_names = os.listdir(data_dir)
		# im_names = [i.strip().split('.')[0] for i in im_names]

		# videoCapture = cv2.VideoCapture('2.avi')
		# success, frame = videoCapture.read() 
		# fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
		# size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
		
		# frame = plt.imread(data_dir+'/000001.png')
		print frame.shape

		fourcc = cv2.VideoWriter_fourcc(*'XVID')  
		videoWriter = cv2.VideoWriter(result_name, fourcc, 10.0, (frame.shape[1], frame.shape[0]))


		for n, im_name in enumerate(im_names):

			im_prefix = im_name.strip().split('.')[0]
			im_path = os.path.join(data_dir, im_name)

			try:
				dets = det_result[int(im_prefix)]
			except:
				dets = []

			# print len(dets)

			im = Image.open(im_path)
			draw = ImageDraw.Draw(im)
			for bbox in dets:
				draw.rectangle([float(i) for i in bbox[1:5]], outline=(0,0,255))  #fill=(255,0,0)  outline


			# tmp = np.asarray(im)
			videoWriter.write(np.asarray(im)[:,:,::-1]) # rgb 2 bgr


			if n%50 ==0:
				print n



if __name__ == '__main__':



	data_dir = './image_02/0001'
	det_file = './0001.txt'

	result_name = 'result.avi'


	show_detection_video(data_dir, det_file, result_name)









