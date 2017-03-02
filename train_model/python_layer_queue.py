import os
import sys
sys.path.insert(0, 'D:/WenyuLv/caffe-windows/caffe-master/Buildx/x64/Release/pycaffe/')
import caffe
import random
from skimage import io, transform
import numpy as np
import logging
import time
from multiprocessing import Pool, Queue, Process #when batch_size is very large
import codecs

class data_layer(caffe.Layer):

#	def _image_preprocess(self, img):
#
#		img = img - [123.0, 117.0, 104.0]
#
#		if self.random_crop and self.phase:
#
#			h_off = max(0, random.randint(0, img.shape[0] - self.random_crop))
#			w_off = max(0, random.randint(0, img.shape[1] - self.random_crop))
#
#			if h_off+self.random_crop< img.shape[0] and w_off+self.random_crop< img.shape[1]:
#				img = img[h_off: h_off+self.random_crop, w_off: w_off+self.random_crop, :]
#			else:
#				hc, wc = int(img.shape[0]/2), int(img.shape[1]/2)
#				img = img[hc-112: hc+112, wc-112: wc+112, :]
#
#
#		if self.mirror and random.random()<0.5 and self.phase:
#			img = img[:, ::-1, :]
#
#		img = img[:,:,[2,1,0]]
#		img = np.transpose(img, [2, 0, 1])
#		
#		#print img.shape
#		return img
#
#	def _load_next_batch(self):
#
#
#		if self._cur+self.batch_size >= len(self._lines):
#			self._cur = 0
#			if self.shuffle: self._perm = np.random.permutation(np.arange(len(self._lines)))
#
#		data = []
#		label = []
#
#		for i in self._perm[self._cur: self._cur+ self.batch_size]:
#			
#			path, lab = self._lines[i].strip().split('\t') # 
#
#			try:
#
#				img = io.imread(path)
#
#			except:
#
#				print 'load xxx',
#				continue
#
#			if self.phase==1 and img.shape[0]>750 and img.shape[1]>750 :
#
#				img = transform.rescale(img, 0.3)*255.0
#				img = self._image_preprocess(img)
#				data.append(img)
#				label.append(int(lab))
#
#			if self.phase==0:
#
#				img = transform.resize(img, [224,224])*255.0
#				img = self._image_preprocess(img)
#				data.append(img)
#				label.append(int(lab))
#
#
#
#		self._cur += self.batch_size
#		
#		return np.array(data), np.array(label)
#

	def setup(self, bottom, top):

		self.layer_params = eval(self.param_str)
		self.data_file = self.layer_params['data_file']

		#self.layer_params['batch_size'] = 100
		self.batch_size = self.layer_params['batch_size']

		self.layer_params['shuffle'] = 1
		self.shuffle = self.layer_params['shuffle']

		self.layer_params['mirror'] = 1
		self.mirror = self.layer_params['mirror']
		
		self.layer_params['crop_size'] = 224
		self.random_crop = self.layer_params['crop_size']
		
		self.layer_params['phase'] = 1
		self.phase = self.layer_params['phase']


		top[0].reshape(3, 3, 224, 224)
		top[1].reshape(1, 1)

		#print self.data_file[0]

		self._lines = []
		for file_name in self.data_file:
			with codecs.open(file_name, 'r', 'utf-8') as f:
				self._lines += f.readlines()


		print 'Total images: {}.'.format(len(self._lines))

		self._cur = 0
		self._perm = np.random.permutation(np.arange(len(self._lines)))

		print 'queue init'

		self.fetcher_processes = []
		
		if self.phase==1:

			self.queue = Queue(3)
			for i in xrange(12):
				self.fetcher_processes.append( Fetcher(self.queue, self._lines, self.layer_params) )
				self.fetcher_processes[i].start()
		
		else: ##
			self.queue = Queue(2)
			for i in xrange(4):
				self.fetcher_processes.append( Fetcher(self.queue, self._lines, self.layer_params) )
				self.fetcher_processes[i].start()

		def clean_up():
			print 'terminatring process'
			for iterm in self.fetcher_processes:
				iterm.terminate()
				iterm.join()
		import atexit
		atexit.register(clean_up)

#	def _process_run(self):
#
#		while True:
#			data, label = self._load_next_batch()
#			blob = {}
#			blob['data'] = data
#			blob['label'] = label
#			self._queue.put(blob)

	def forward(self, bottom, top):

		#print 'forward'

		start = time.time()


		blob = self.queue.get()
		data = blob['data']
		label = blob['label']

		#data, label = self._load_next_batch()
		#print '------data.shape: {}'.format(data.shape)


		top[0].reshape(*(data.shape))
		top[0].data[...] = data

		top[1].reshape(*(label.shape))
		top[1].data[...] = label

		#print data.shape, label.shape

		#print time.time()-start
		#logging.warning('-----------------{}'.format(time.time()-start))

	def reshape(self, bottom, top):
		pass

	def backward(self, top, propagate_down, bottom):
		pass


	#def prefetch(self):
	#	self.queue = Queue(5)
	#	fetcher_process = Fetcher(self.queue, self._lines, self.layer_params)
	#	fetcher_process.start()


class Fetcher(Process):
	"""docstring for Fetcher"""
	def __init__(self, queue, lines, params):
		super(Fetcher, self).__init__()

		self._queue = queue

		self._lines = lines

		self.batch_size = params['batch_size']
		self.shuffle = params['shuffle']
		self.mirror = params['mirror']
		self.random_crop = params['crop_size']
		self.phase = params['phase']

		self._cur = 0
		self._perm = np.random.permutation( np.arange(len(self._lines))[::3] ) #

		

	def _image_preprocess(self, img):

		img = img - [123.0, 117.0, 104.0]

		if self.phase:

			if self.random_crop:

				h_off = max(0, random.randint(0, img.shape[0] - self.random_crop+1))
				w_off = max(0, random.randint(0, img.shape[1] - self.random_crop+1))

				if h_off+self.random_crop< img.shape[0] and w_off+self.random_crop< img.shape[1]:
					img = img[h_off: h_off+self.random_crop, w_off: w_off+self.random_crop, :]
				else:
					hc, wc = int(img.shape[0]/2), int(img.shape[1]/2)
					img = img[hc-112: hc+112, wc-112: wc+112, :]

			if self.mirror and random.random()<0.4:

				img = img[:, ::-1, :]

		img = img[:,:,[2,1,0]]
		img = np.transpose(img, [2, 0, 1])

		return img

	def _load_next_batch(self):

			if self._cur+self.batch_size >= len(self._lines)/3:
				self._cur = 0
				if self.shuffle: self._perm = np.random.permutation( np.arange(len(self._lines))[::3] ) #

			data = []
			label = []

			for i in self._perm[self._cur: self._cur+ self.batch_size]:
				
				for j in range(3):

					path, lab = self._lines[i+j].strip().split('\t') # 
					try:
						img = io.imread(path)
					except:
						print 'load xxx',
						break
						#continue

					if self.phase==1 :
		
						if min(img.shape) > 224:
							img = transform.rescale(img, 224.0/min(img.shape)+0.02 )*255.0

						else :
							img = transform.resize(img, [256, 256])*255.0

						img = self._image_preprocess(img)

						data.append(img)
						label.append(int(lab))

					elif self.phase==0 :

						img = transform.resize(img, [224,224])*255.0
						img = self._image_preprocess(img)
						data.append(img)
						label.append(int(lab))


			self._cur += self.batch_size

			#assert len(data) == len(label)

			data = data[0::3]+data[1::3]+data[2::3]
			label = label[0::3]

			#print '--------------------'
			#print len(data)


			assert len(data) == len(label)*3
			
			return np.array(data), np.array(label)


	def _get_batch_blob(self):

		#print '----------------------------------------------------------fetch-1'
		
		data, label = self._load_next_batch()
		blob = {}
		blob['data'] = data
		blob['label'] = label

		#print '----------------------------------------------------------fetch-2'

		return blob

	def run(self):

		#print '----------------------------------------------------------fetch'
		while True:

			blob = self._get_batch_blob()
			self._queue.put(blob)


