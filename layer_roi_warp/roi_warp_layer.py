import sys
sys.path.insert(0, '')
import caffe
import numpy as np 
from skimage import transform


class roi_warp_layer(caffe.Layer):

	def setup(self,bottom, top):
		
		params = eval(self.param_str)
		self.out_size = params['out_size']
		self.spatial_scale = 0

		top[0].reshape(bottom[1].shape[0], 1, self.out_size, self.out_size)

	def reshape(self, bottom, top):
		pass

	def forward(self, bottom, top):
		
		feature_map = bottom[0].data
		rois = bottom[1].data

		rois_nums = rois.shape[0]
		fn, fc, fh, fw = feature_map.shape
		assert fn==1

		roi_outputs = np.zeros([rois_nums, fc, self.out_size, self.out_size])
		
		for i in range(rois_nums):
			#rois[i] and feature_map ==> roi_output[i, ...]
			y, x, h, w = rois[i]

			cropped_fea = feature_map[0, :, 3:3+6, 3:3+4] #cropped roi_fea
			cropped_fea = np.transpose(cropped_fea, (1,2,0))

			warped_fea = transform.resize(cropped_fea, (self.out_size, self.out_size))
			warped_fea = np.transpose(warped_fea, (2,0,1))

			roi_outputs[i, ...] = warped_fea

		top[0].reshape(*(roi_outputs.shape))
		top[0].data[...] = roi_outputs

	def backward(self, top, propagate_down, bottom):
		#print top[0].diff.shape
		#print bottom[0].diff.shape
		#print bottom[1].diff.shape
		#print bottom[1].data.shape
		feature_map = bottom[0].data
		rois = bottom[1].data

		bottom[0].diff = 0

		for i in range(top[0].diff.shape[0]):
			
			bbox = top[1].data[i, ...]
			
			h_shift = bbox[]
			w_shift = bbox[]
			ih = bbox[]
			iw = bbox[]

			coor_out, coor_fea, delta_h, delta_w = self._bilinear_transpose(ih, iw, self.out_size, self.out_size)

			for c in top[0].diff.shape[1]:

				for k in range(len(coor_out)):

					diff_top = top[0].diff[i, c, coor_out[k][0], coor_out[k][1]]

					bottom[0, c, h_shift+ coor_fea[k][0],   w_shift+ coor_fea[k][1]] += (1-delta_h)* (1-delta_w)* diff_top
					bottom[0, c, h_shift+ coor_fea[k][0]+1, w_shift+ coor_fea[k][1]] += delta_h * (1-delta_w)* diff_top
					bottom[0, c, h_shift+ coor_fea[k][0],   w_shift+ coor_fea[k][1]+1] += (1-delta_h)* delta_w* diff_top
					bottom[0, c, h_shift+ coor_fea[k][0]+1, w_shift+ coor_fea[k][1]+1] += delta_h * delta_w* diff_top


	def _bilinear_transpose(self, ih, iw, oh, ow)


		sh = ih / oh
		sw = iw / ow

		ho, wo = np.meshgrid(np.arange(self.out_size), np.arange(self.out_size))
		h = floor(ho * sh)
		w = floor(wo * sw)

		h = np.maximum(h, 0)
		w = np.maximum(w, 0)

		h = np.minimum(h, ih - 2)
		w = np.minimum(w, iw - 2)

		delta_h = ho * sh - h
		delta_w = wo * sw - w 

		coor_fea = np.vstack(h.flatten(), w.flatten()).transpose()
		coor_out = np.vstack(ho.flatten(), wo.flatten()).transpose()

		return coor_out, coor_fea, delta_h, delta_w