import sys
sys.path.insert(0, '../../caffe-windows/Build/x64/Debug/pycaffe/')
import caffe
import numpy as np
import sympy


class data_layer(caffe.Layer):
	def setup(self, bottom, top):
		self.top_names = ['data', 'target']

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		pass

	def backward(self, top, propagate_down, bottom):
		pass



class loss_layer(caffe.Layer):
	
	def setup(self, bottom, top):
		if len(bottom) != 2:
			raise Exception(" need two inputs to compute loss")

	def reshape(self, bottom, top):
		
		self.diff = np.zeros_like( bottom[0].data, dtype= np.float32)
		top[0].reshape(1)

	def forward(self, bottom, top):

		def get_coef(output):
			z = output.T
			z_mux, z_muy, z_sx, z_sy, z_corr = z # make sure output shape 5*1
			z_sx, z_sy= np.exp(z_sx), np.exp(z_sy)  # dev must non-negtive
			z_corr = np.tanh(z_corr) # [-1 1]
			return [z_mux, z_muy, z_sx, z_sy, z_corr]

		def prob_2d_gaussian(x,y, mux, muy, sx, sy, rho):
			normx, normy = x-mux, y-muy
			sxsy = sx*sy
			z = np.square(normx/sx) + np.square(normy/sy) - 2.0*rho*(normx*normy) / (sxsy)
			negRho = 1.0 - np.square(rho)
			result0 = np.exp(-z / 2.0 / negRho) #/ ( 2 * np.pi * sxsy * np.sqrt(negRho) )
			denom = 2 * np.pi * sxsy * np.sqrt(negRho)
			result = result0 / denom
			self.result = result
			self.term0, self.term1 = denom, result0
			self.normx, self.normy = normx, normy
			self.sxsy = sxsy
			self.z = z
			self.negRho = negRho
			return result

		def get_lossfunc(z_mux, zmuy, z_sx, z_sy, z_corr, x_data, y_data):
			result0 = prob_2d_gaussian(x_data, y_data, z_mux, zmuy, z_sx, z_sy, z_corr)
			epsilon = 1e-20
			result1 = -np.log(np.max(result0, epsilon))
			return np.sum(result1)

		[o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(bottom[0].data)
		loss = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, bottom[1].data[0], bottom[1].data[1])
		top[0].data[...] = loss
		# stor the network result, result/p
		self.loss = loss
		self.mux = o_mux
		self.muy = o_muy
		self.sx = o_sx
		self.sy = o_sy
		self.corr = o_corr

	def backward(self, top, propagate_down, bottom):
		if propagate_down[0]:
			deltaP = -( 1.0 / (self.result+1e-5) )

			delta_mux = (-2*self.normx / self.sx**2) + (2*self.corr*self.normy/self.sxsy) / (-2*self.negRho) 
			bottom[0].diff[0,0] = deltaP * (self.term0 * self.term1 * delta_mux) * 1.0

			delta_muy = (-2*self.normy / self.sy**2) + (2*self.corr*self.normx/self.sxsy) / (-2*self.negRho)
			bottom[0].diff[0,1] = deltaP * (self.term0 * self.term1 * delta_muy) * 1.0

			delta_sx0 = self.term1 / (-2*np.pi*self.sxsy*self.sy*np.sqrt(self.negRho))
			delta_sx1 = self.term0 * self.term1 * ((self.normx**2*self.sx/self.sx**4 + self.corr*self.normx*self.normy*self.sy/self.sxsy**2) / self.negRho)
			bottom[0].diff[0,2] = deltaP * (delta_sx0 + delta_sx1) * self.sx

			delta_sy0 = self.term1 / (-2*np.pi*self.sxsy*self.sx*np.sqrt(self.negRho))
			delta_sy1 = self.term0 * self.term1 * ((self.normy**2*self.sy/self.sy**4 + self.corr*self.normx*self.normy*self.sx/self.sxsy**2) / self.negRho)
			bottom[0].diff[0,3] = deltaP * (delta_sy0 + delta_sy1) * self.sy

			delta_corr0 = (self.corr*self.negRho**(-3.0/2)/(np.pi*self.sxsy)) * self.term1
			delta_corr1 = self.term0 * self.term1 * (-self.corr/self.negRho**2*self.z + self.corr*self.normy*self.normx/(self.sxsy*self.negRho))
			bottom[0].diff[0,4] = deltaP * (delta_corr0 + delta_corr1) * (1.0-self.corr**2)

	