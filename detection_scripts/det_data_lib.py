import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid

from fast_rcnn.config import cfg

import glob


class kitti(imdb):

    def __init__(self, image_set):
        imdb.__init__(self, image_set)

        self._image_set = image_set
        
        self._data_path = 'image_02/'
        self._label_path = 'label_02/'

        #self._classes = ('__background__', 'Cyclist', 'Car', 'Pedestrian') ['Van', 'Car']
        self._car_type = ['van', 'car']
        self._classes = ('__background__', 'car')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        #self._image_ext = '.png'

        #self._num_per_folder = self._get_num_per_folder()

        self._images, self._fram_data = self._get_images()
        self._image_index = range(len(self._images)) #self._load_image_set_index()
        
        


    def image_path_at(self, i):

        return self.image_path_from_index( self._image_index[i] )  ####

    def image_path_from_index(self, index):

        return self._images[index]


    def gt_roidb(self):

        gt_roidb = [ self._load_kitti_annotation(k) for k in self._fram_data]

        assert len(gt_roidb) == len(self._image_index)

        return gt_roidb




    def _load_kitti_annotation(self, fram_data):


        num_objs = len(fram_data)

        
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # i = 0
        for i, lin in enumerate(fram_data):

            if lin[2].lower() in self._car_type:
                cls_ind = 1
                overlaps[i, cls_ind] = 1.0

                gt_classes[i] = cls_ind

            elif lin[2] == 'DontCare':
                cls_ind = 0
                # overlaps[i, cls_ind] = 0.0

                gt_classes[i] = cls_ind
                
            # else:
            #     print 'cls ===== cls'

            x1 = max(0., float(lin[6])-1.0)  #x1 y1 x2 y2
            y1 = max(0., float(lin[7])-1.0)
            x2 = max(0., float(lin[8])-1.0)
            y2 = max(0., float(lin[9])-1.0)

            # gt_classes[i] = cls_ind
            # overlaps[i, 1] = 1.0
            boxes[i, :] = [x1, y1, x2, y2]
            seg_areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1)

        # boxes = np.array(boxes)
        # assert (boxes[:, 2] >= boxes[:, 0]).all()


        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes':boxes,'gt_classes':gt_classes,'gt_overlaps':overlaps,\
                'flipped':False,'seg_areas':seg_areas}



    # def _load_image_set_index(self):

    #     return range(len(self._images))


    # def _get_num_per_folder(self):

    #     nums = [len(glob.glob(folder+'/*.png')) for folder in glob.glob(self._data_path + '/*')]
    #     return nums


        
    def _get_images(self):

        img_dirs = os.listdir(self._data_path)
        gt_dirs = os.listdir(self._label_path)

        fram_data = {}
        neg_data = {}

        for img_path, gt_path in zip(img_dirs, gt_dirs):

            imgs = glob.glob(self._data_path+img_path+'/*.png')
            
            with open(self._label_path+gt_path) as f:
                
                gt_raw_data = f.readlines()
                gt_raw_data = [l.strip().split(' ') for l in gt_raw_data]

                for ll in gt_raw_data:
                    
                    if ll[2] in ['Car', 'Van']:
                        try:
                            fram_data[imgs[int(ll[0])]].append(ll)
                        except:
                            fram_data[imgs[int(ll[0])]] = [ll]

                    elif ll[2] == 'DontCare' :

                        try:
                            neg_data[imgs[int(ll[0])]].append(ll)
                        except:
                            neg_data[imgs[int(ll[0])]] = [ll]



        new_keys = list( set(fram_data.keys()) & set(neg_data.keys()) ) 

        final_data = {}

        for k in new_keys:

            final_data[k] = fram_data[k] + neg_data[k]

        return final_data.keys(), final_data.values()
