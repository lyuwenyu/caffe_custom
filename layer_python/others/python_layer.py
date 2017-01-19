import os
import sys
sys.path.insert(0, '../../caffe-windows/Build/x64/Debug/pycaffe/')
import caffe
import random
from skimage import io, transform
import numpy as np
import logging
import time
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

class data_layer(caffe.Layer):

	def _preprocess_image(self, img):
		mean_pix = [103.939, 116.779, 123.68]
		if self.new_height> 0 and self.new_width >0:
			img = transform.resize(img, (self.new_height, self.new_width))
			#TODO bbx
		if self.mirror and random.random>0.5:
			img = img[:,::-1,:]
		if self.crop_size> 0:
			pass
		img = img[:,:,[2,1,0]]
		img = img.transpose([2,0,1])
		for i in xrange(3):
			img[i,:,:] = img[i,:,:] - mean_pix[i]
		#img = img - np.array(mean_pix).reshape(3,1,1)
		return img

	def _load_data(self, path):
		#im = caffe.io.load_image(path)
		im = io.imread(path) * 1.0
		return self._preprocess_image(im)

	def _shuffle(self):
		self._cur = 0
		self._perm = np.random.permutation(np.arange(len(self._lines)))

	def _precess_bbx(self, bbx):
		pass

	def _load_per_func(self, idx):
		#logging.warning('{}'.format(idx))
		info = self._lines[idx].strip().split(' ')
		return (self._load_data(info[0]), info[1], info[2:])

	def _load_next_batch(self):

		#random.shuffle(self._lines)
		if self._cur+self.batch_size>=len(self._lines):
			self._cur = 0
			if self.shuffle: self._perm = np.random.permutation(np.arange(len(self._lines)))
				
		#data, label, bbx = [], [], []
		#paths = []
		#for line in self._lines[self._cur: self._cur+self.batch_size]:

		#for idx in self._perm[self._cur: self._cur+self.batch_size]:
		#	info = self._lines[idx].strip().split(' ')
		#	#info = line.strip().split(' ')
		#	#paths.append(info[0])
		#	data.append(self._load_data(info[0]))
		#	label.append(info[1])
		#	bbx.append(info[2:])

			
		#if self.batch_size > 50:
		#	pool = ThreadPool(8)
		#	data = pool.map(self._load_data, paths) #TODO which one return first
		#	pool.close()
		#	pool.join()
		#else :
		#	data = map(self._load_data, paths)
		pool = ThreadPool(8)
		res = pool.map(self._load_per_func, self._perm[self._cur: self._cur+self.batch_size])
		pool.close()
		pool.join()
		data = np.array([item[0] for item in res])
		label = np.array([item[1] for item in res])

		bbx = np.zeros((self.batch_size, 5))
		for i, item in enumerate(res):
			bbx[i,0] = i
			bbx[i,1:] = item[2]
		
		#bbx[:,0] = np.arange(self.batch_size)

		#bbx = np.array([item[2] for item in res])
		#batch_idx = np.arange(self.batch_size).reshape(self.batch_size,1)
		#bbx = np.hstack((batch_idx, bbx))

		self._cur+=self.batch_size

		return data, label, bbx

	def setup(self, bottom, top):
		layer_params = eval(self.param_str)
		self.data_file = layer_params['data_file']
		self.batch_size = layer_params['batch_size']
		self.new_height = layer_params['new_height']
		self.new_width = layer_params['new_width']
		self.shuffle = layer_params['shuffle']
		self.mirror = layer_params['mirror']
		self.crop_size = 0

		self._name_to_top_map = {}
		idx = 0
		top[idx].reshape(self.batch_size, 3, 1, 1) #self.new_height, self.new_width)
		self._name_to_top_map['data'] = idx
		idx+=1

		top[idx].reshape(self.batch_size, )
		self._name_to_top_map['label'] = idx
		idx+=1

		top[idx].reshape(self.batch_size, 5)
		self._name_to_top_map['rois'] = idx

		with open(self.data_file) as f:
			self._lines = f.readlines()

		#self.batchs_per_epoch = int(len(self.lines) / self.batch_size)
		self._cur = 0
		self._perm = np.arange(len(self._lines))

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):

		start = time.time()

		data, label, bbox = self._load_next_batch()
		#self.cur+=1
		#logging.warning('{} {} {}'.format(np.array(data).shape, np.array(label).shape, np.array(bbox).shape))
		#print np.array(data).shape, np.array(label).shape, np.array(bbox).shape
		top[0].reshape(*(data.shape))
		top[0].data[...] = data
		top[1].data[...] = label
		top[2].data[...] = bbox

		logging.warning('{}'.format(time.time()-start))

	def backward(self, top, propagate_down, bottom):
		pass


class DataLoader():
	def __init__(self):
		pass