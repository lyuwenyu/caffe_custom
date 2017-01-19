import os
import sys
sys.path.insert(0, '../../../caffe-windows/Build/x64/Debug/pycaffe/')
import caffe
import random
from skimage import io, transform
import numpy as np
import logging
import time
#from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import Pool #when batch_size is very large
import codecs

class data_layer(caffe.Layer):

	def _image_preprocess(self, img):

		img = transform.rescale(img, 0.3)*255.0
		img = img - [104.0, 117.0, 123.0]

		if self.random_crop:

			hc, wc = int(img.shape[0]/2), int(img.shape[1]/2)
			try:
				shift = np.random.choice([-30, -15, 0, 15, 30], size=2, p=[0.2, 0.2, 0.2, 0.2, 0.2])
				hcn , wcn= hc+shift[0], wc+shift[1]
				img = img[hcn-112: hcn+112, wcn-112: wcn+112, :] #self._crop_size
			except:
				img = img[hc-112: hc+112, wc-112: wc+112, :]

		if self.mirror and random.random()<0.5:
			img = img[:, ::-1, :]

		img = img[:,:,[2,1,0]]
		img = np.transpose(img, [2, 0, 1])
		
		return img

	def _load_next_batch(self):

		if self._cur+self.batch_size >= len(self._lines):
			self._cur = 0
			if self.shuffle: self._perm = np.random.permutation(np.arange(len(self._lines)))

		data = []
		label = []
		for i in self._perm[self._cur: self._cur+ self.batch_size]:
			
			path, lab = self._lines[i].strip().split('\t')
			
			try:
				img = io.imread(path)
			except:
				print 'load xxx'
				continue
			if img.shape[0]>750 and img.shape[1]>750:

				img = self._image_preprocess(img)

				data.append(img)
				label.append(int(lab))
			

		self._cur += self.batch_size
		
		return np.array(data), np.array(label)

	def setup(self, bottom, top):
		layer_params = eval(self.param_str)
		self.data_file = layer_params['data_file']
		self.batch_size = 10 #layer_params['batch_size']
		self.shuffle = 1 #layer_params['shuffle']
		self.mirror = 1 #layer_params['mirror']
		self.random_crop = 1


		top[0].reshape(1, 3, 224, 224)
		top[1].reshape(1, 1)


		with codecs.open(self.data_file, 'r', 'utf-8') as f:
			self._lines = f.readlines()

		print len(self._lines)
		random.shuffle(self._lines)
		#self.batchs_per_epoch = int(len(self.lines) / self.batch_size)
		self._cur = 0
		self._perm = np.arange(len(self._lines))



	def forward(self, bottom, top):

		#start = time.time()

		data, label= self._load_next_batch()
		
		#print 'xxxxxxxx'
		top[0].reshape(*(data.shape))
		top[0].data[...] = data

		top[1].reshape(*(label.shape))
		top[1].data[...] = label


		#logging.warning('{}'.format(time.time()-start))

	def reshape(self, bottom, top):
		pass
	def backward(self, top, propagate_down, bottom):
		pass

