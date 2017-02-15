import os
import sys
sys.path.insert(0, 'D:/WenyuLv/caffe-windows/caffe-master/Buildx/x64/Release/pycaffe/')
import caffe
import random
from skimage import io, transform
import numpy as np
import logging
import time
#from multiprocessing import Pool #when batch_size is very large
from multiprocessing.dummy import Pool as ThreadPool
import codecs

class data_layer(caffe.Layer):

	def _image_preprocess(self, img):

		img = img - [123.0, 117.0, 104.0]

		if self.random_crop and self.phase:

			h_off = max(0, random.randint(0, img.shape[0] - self.random_crop))
			w_off = max(0, random.randint(0, img.shape[1] - self.random_crop))

			if h_off+self.random_crop< img.shape[0] and w_off+self.random_crop< img.shape[1]:
				img = img[h_off: h_off+self.random_crop, w_off: w_off+self.random_crop, :]
			else:
				hc, wc = int(img.shape[0]/2), int(img.shape[1]/2)
				img = img[hc-112: hc+112, wc-112: wc+112, :]


		if self.mirror and random.random()<0.5 and self.phase:
			img = img[:, ::-1, :]

		img = img[:,:,[2,1,0]]
		img = np.transpose(img, [2, 0, 1])
		
		return img

	def per_img_process(self, line):

		path, lab = line.strip().split('\t') # 

		try:
			img = io.imread(path)
		except:
			print 'load xxx',
			return np.zeros([3, self.random_crop, self.random_crop]), -1

		if self.phase==1 :
			if img.shape[0]>=1000 and img.shape[1]>=1000 :
				img = transform.rescale(img, 0.23)*255.0  ####

			elif img.shape[0]>=900 and img.shape[1]>=900 :
				img = transform.rescale(img, 0.25)*255.0  ####

			elif img.shape[0]>=750 and img.shape[1]>=750 :
				img = transform.rescale(img, 0.3)*255.0  ####

			else :
				img = transform.resize(img, [256, 256])*255.0

			img = self._image_preprocess(img)
			return img, int(lab)

		elif self.phase==0:

			img = transform.resize(img, [self.random_crop, self.random_crop])*255.0
			img = self._image_preprocess(img)
			return img, int(lab)

		else:
			return np.zeros([3, self.random_crop, self.random_crop]), -1



	def _load_next_batch(self):


		if self._cur+self.batch_size >= len(self._lines):
			self._cur = 0
			if self.shuffle: self._perm = np.random.permutation(np.arange(len(self._lines)))

		#pool = mp.ProcessingPool(32)
		#pool = ThreadPool(8)
		pool = Pool(8)

		iterms = pool.map(self.per_img_process, [ self._lines[i] for i in self._perm[self._cur: self._cur+ self.batch_size] ])

		pool.close()
		pool.join()

		self._cur += self.batch_size

		data = np.array([iterm[0] for iterm in iterms])
		label = np.array([iterm[1] for iterm in iterms])

		mask = np.where( label != -1 )

		#print iterms[0][0].shape
		#print data.shape
		#print label.shape
		#print data[mask].shape
		#print label[mask].shape

		return data[mask], label[mask]


	def setup(self, bottom, top):

		layer_params = eval(self.param_str)
		self.data_file = layer_params['data_file']
		self.batch_size = layer_params['batch_size']
		self.shuffle = 1 #layer_params['shuffle']
		self.mirror = 1 #layer_params['mirror']
		
		self.random_crop = 224 #layer_params['crop_size']
		self.phase = layer_params['phase']

		top[0].reshape(1, 3, 224, 224)
		print top[0].data.shape

		top[1].reshape(1, 1)

		with codecs.open(self.data_file, 'r', 'utf-8') as f:
			self._lines = f.readlines()

		print 'Total images: {}.'.format(len(self._lines))

		self._cur = 0
		self._perm = np.random.permutation(np.arange(len(self._lines)))



	def forward(self, bottom, top):

		start = time.time()
		#logging.warning('{}'.format('xxxxxxx'))

		data, label= self._load_next_batch()
		#print data.shape, label.shape
		
		top[0].reshape(*(data.shape))
		top[0].data[...] = data

		top[1].reshape(*(label.shape))
		top[1].data[...] = label


		#logging.warning('------{}'.format(time.time()-start))

	def reshape(self, bottom, top):
		pass

	def backward(self, top, propagate_down, bottom):
		pass

