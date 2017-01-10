import sys
sys.path.insert(0, '../../caffe-windows/Build/x64/Debug/pycaffe/')
import caffe
import numpy as np
#import sympy  #calculate gradien
import theano
import theano.tensor as T

import os
import pickle
import random
import logging

class data_layer(caffe.Layer):
	def setup(self, bottom, top):
		self.top_names = ['coordinate', 'target', 'cont']
		self.params = eval(self.param_str)
		self.params['data_dirs'] = ['D:\\workspace\\social-lstm-tf\\data\\eth\\univ']

		self.data_loader = BatchLoader(self.params)
		self.data_loader.reset_batch_pointer()

		top[0].reshape(self.params['batch_size'], self.params['seq_length'], 2)
		top[1].reshape(self.params['batch_size'], self.params['seq_length'], 2)
		top[2].reshape(self.params['batch_size'], self.params['seq_length'])

	def forward(self, bottom, top):
		x, y = self.data_loader.next_batch()
		top[0].data[...] = x
		top[1].data[...] = y
		top[2].data[...] = 1
		top[2].data[0,0] = 0

	def reshape(self, bottom, top):
		pass
	def backward(self, top, propagate_down, bottom):
		pass

class BatchLoader(object):
	def __init__(self, params):
		self.batch_size = params['batch_size']
		self.data_dirs = params['data_dirs']
		self.seq_length = params['seq_length']
		self.data_dir = '.'
		data_file = os.path.join(self.data_dir, '{}_trajectories.cpkl'.format('xx'))
		if not(os.path.exists(data_file)):
			self.preprocess(self.data_dirs, data_file)

		self.load_preprocessed(data_file)
		self.reset_batch_pointer()

	def preprocess(self, data_dirs, data_file):
		current_ped = 0
		all_ped_data = {}
		dataset_indices = []
		for directory in data_dirs:
			file_path = os.path.join(directory, 'pixel_pos.csv')
			data = np.genfromtxt(file_path, delimiter=',')
			numPeds = np.size(np.unique(data[1,:]))
			for ped in range(1, numPeds+1):
				traj = data[:, data[1,:] == ped]
				traj = traj[[3,2,0], :]
				all_ped_data[current_ped+ped] = traj
			dataset_indices.append(current_ped+numPeds)
			current_ped += numPeds
		complete_data = (all_ped_data, dataset_indices)
		with open(data_file, 'wb') as f:
			pickle.dump(complete_data, f, protocol=2)

	def load_preprocessed(self, data_file):
		with open(data_file, 'rb') as f:
			self.raw_data = pickle.load(f)
		all_ped_data = self.raw_data[0]

		self.data = []
		counter = 0
		for ped in all_ped_data:
			traj = all_ped_data[ped]
			if traj.shape[1] > self.seq_length+2:
				self.data.append(traj[[0,1], :].T)
				counter += int(traj.shape[1] / ((self.seq_length+2)))
		self.num_batches = int( counter / self.batch_size )

	def next_batch(self):
		x_batch = [] #train
		y_batch = [] #target
		for i in xrange(self.batch_size):
			traj = self.data[self.pointer]
			n_batch = int(traj.shape[0] / (self.seq_length+2))
			idx = random.randint(0, traj.shape[0] - self.seq_length - 2)
			x_batch.append(np.copy(traj[idx:idx+self.seq_length, :]))
			y_batch.append(np.copy(traj[idx+1:idx+self.seq_length+1, :]))

			if random.random() < (1.0/float(n_batch)):
				self.tick_batch_pointer()
		return x_batch, y_batch

	def reset_batch_pointer(self):
		self.pointer = 0

	def tick_batch_pointer(self):
		self.pointer+=1
		if (self.pointer >= len(self.data)):
			self.pointer = 0



class loss_layer(caffe.Layer):
	
	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception(" need two inputs to compute loss")

		x,y = T.dvectors('x', 'y')
		mux, muy = T.dvectors('mux', 'muy')
		sx, sy = T.dvectors('sx', 'sy')
		rho = T.dvector('rho')

		z_sx = T.exp(sx)
		z_sy = T.exp(sy)
		z_corr = T.tanh(rho)

		z = T.square((x-mux)/z_sx) + T.square((y-muy)/z_sy) - 2.0*z_corr*(x-mux)*(y-muy)/(z_sx*z_sy)
		prob = T.exp( -z/(2.0*(1.0-T.square(z_corr))) ) / ( 2.0*np.pi*z_sx*z_sy*T.sqrt(1.0-T.square(z_corr)) )

		result = T.sum(-T.log(T.maximum(prob, 1e-10)))

		#dmux, dmuy, ds, ds, drho = T.grad(result, [mux,muy,sx,sy,rho])
		dmux = T.grad(result, mux)
		dmuy = T.grad(result, muy)
		dsx = T.grad(result, sx)
		dsy = T.grad(result, sy)
		drho = T.grad(result, rho)

		self.f = theano.function([mux, muy, sx, sy, rho, x, y], result)
		self.dfmux = theano.function([mux, muy, sx, sy, rho, x, y], dmux)
		self.dfmuy = theano.function([mux, muy, sx, sy, rho, x, y], dmuy)
		self.dfsx = theano.function([mux, muy, sx, sy, rho, x, y], dsx)
		self.dfsy = theano.function([mux, muy, sx, sy, rho, x, y], dsy)
		self.dfrho = theano.function([mux, muy, sx, sy, rho, x, y], drho)

	def reshape(self, bottom, top):
		
		self.diff = np.zeros_like(bottom[0].data, dtype= np.float32)
		top[0].reshape(1)

	def forward(self, bottom, top):

		bottom0 = np.array(bottom[0].data)
		bottom1 = np.array(bottom[1].data)
		imux, imuy, isx, isy, irho = bottom0.T
		ix, iy = bottom1.T
		imux, imuy = imux.reshape(bottom0.shape[0]), imuy.reshape(bottom0.shape[0])
		isx, isy = isx.reshape(bottom0.shape[0]), isy.reshape(bottom0.shape[0])
		irho= irho.reshape(bottom0.shape[0]) 
		ix, iy = ix.reshape(bottom0.shape[0]), iy.reshape(bottom0.shape[0])

		top[0].data[...] = self.f(imux, imuy, isx, isy, irho, ix, iy)

	def backward(self, top, propagate_down, bottom):
		if propagate_down[0]:
			bottom0 = np.array(bottom[0].data)
			bottom1 = np.array(bottom[1].data)
			imux, imuy, isx, isy, irho = bottom0.T
			ix, iy = bottom1.T
			imux, imuy = imux.reshape(bottom0.shape[0]), imuy.reshape(bottom0.shape[0])
			isx, isy = isx.reshape(bottom0.shape[0]), isy.reshape(bottom0.shape[0])
			irho= irho.reshape(bottom0.shape[0]) 
			ix, iy = ix.reshape(bottom0.shape[0]), iy.reshape(bottom0.shape[0])

			bottom[0].diff[:,0] = self.dfmux(imux, imuy, isx, isy, irho, ix, iy)
			bottom[0].diff[:,1] = self.dfmuy(imux, imuy, isx, isy, irho, ix, iy)
			bottom[0].diff[:,2] = self.dfsx(imux, imuy, isx, isy, irho, ix, iy)
			bottom[0].diff[:,3] = self.dfsy(imux, imuy, isx, isy, irho, ix, iy)
			bottom[0].diff[:,4] = self.dfrho(imux, imuy, isx, isy, irho, ix, iy)
	

class inner_product_layer(caffe.Layer):
	def setup(self, bottom, top):
		if len(bottom) != 1:
			raise Exception(" need two inputs to compute loss")

	def reshape(self, bottom, top):
		self.diff = np.zeros_like( bottom[0].data, dtype= np.float32)
		top[0].reshape(1)
		pass

	def forward(self, bottom, top):
		pass

	def backward(self, top, propagate_down, bottom):
		pass


class sample_layer(caffe.Layer):
	def setup(self, bottom, top):
		pass
	def reshape(self, bottom, top):
		pass
	def forward(self, bottom, top):
		pass
	def backward(self, top, propagate_down, bottom):
		pass

