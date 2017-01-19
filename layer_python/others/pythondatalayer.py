import caffe
import numpy as np

class gene_data(caffe.Layer):

	def setup(self, bottom, top):
		self.top_names = ['data', 'label', 'cont']
		param = eval(self.param_str)
		self.batch_size = param['batch_size']
		self.shuffle = param['shuffle']
		top[0].reshape(self.batch_size, 3, 10,10)
		top[1].reshape(self.batch_size, 1)
		top[2].reshape(self.batch_size, 1)

	def reshape(self, bottom, top):
		pass
		
	def forward(self, bottom, top):
		top[0].data[...] = np.random.randn(self.batch_size, 3, 10,10)
		top[1].data[...] = np.random.randn(self.batch_size, 1)
		top[2].data[...] = np.random.randn(self.batch_size, 1)
	
	def backward(self, top, propagate_down, bottom):
		pass


