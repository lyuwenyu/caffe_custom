#coding=utf-8

#import os  
#import re
#import sys  
import time  
#import chardet  
#import datetime  
import tkFileDialog  
import tkMessageBox  
from   Tkinter import *  

import caffe
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io
from PIL import Image, ImageDraw, ImageFont


car_style_rear = [u'Honda思铂睿2009-2010-2013',
#u'Honda思铂睿2013',
u'Honda思铂睿2015',
u'Honda雅阁2003-2005',
u'Honda雅阁2006-2007',
u'Honda雅阁2008-2013',
u'Honda雅阁2014-2016',
u'Honda雅阁1998-2001',
u'VW帕萨特2000-2005',
u'VW帕萨特2006-2008',
u'VW帕萨特2010',
u'VW帕萨特2011-2016',
u'VW迈腾2007-2010',
u'VW迈腾2011-2017']


img_filename = u''
model_filename = u''
net_filename = u''

net = ''

def get_img_file():  
	global img_filename
	img_filename = tkFileDialog.askopenfilename(filetypes=[("text file", "*.jpg")])  
	var_img.set(img_filename)  


def get_net_file():  
	global net_filename
	net_filename = tkFileDialog.askopenfilename(filetypes=[("text file", "*.prototxt")])  
	var_net.set(net_filename) 

def get_model_file():  
	global model_filename  
	global net

	model_filename = tkFileDialog.askopenfilename(filetypes=[("binary file", "*.caffemodel")])  
	var_model.set(model_filename) 

	if net_filename:
		net = caffe.Net(str(net_filename), str(model_filename), caffe.TEST)

def show_result(info):  
	tkMessageBox.showinfo("result ", info)  


def net_init(net, model):

	net = caffe.Net(net, model, caffe.TEST)

	return net

def image_preprocess(img, size=[224, 224]):

	img = transform.resize(img, size)*255.0
	img = img - [123.0, 117.0, 104.0]
	img = img[:,:,[2,1,0]]
	img = np.transpose(img, [2, 0, 1])
	return np.expand_dims(img, 0)


def run():


	global net
	if not net:
		net = caffe.Net(str(net_filename), str(model_filename), caffe.TEST)

	start = time.time()

	img = plt.imread(img_filename)

	#img = io.imread(data_root + lin_part[0])
	imged1 = image_preprocess(img)

	try:
		img = transform.rescale(img, 224.0/min(img.shape[:2])+0.005)
		off_h = max(0, (img.shape[0] - 224)/2-1)
		off_w = max(0, (img.shape[1] - 224)/2-1)
		img = img[off_h: off_h+224, off_w: off_w+224, :]
		imged2 = image_preprocess(img)

		imged = np.vstack([imged1, imged2])

	except:
		imged = imged1


	net.blobs['data'].reshape(*(imged.shape))
	net.blobs['data'].data[...] = imged

	res = net.forward()['prob']

	if res.shape[0] == 2:
		pre = np.argmax(res, 1)[np.argmax(np.max(res, 1))]
	else:
		pre = np.argmax(res[0])



	info = u'label: {}\n time: {}'.format(car_style_rear[pre], time.time()-start)

	var_label.set(info)
	#info = '{}\n {}\n'.format(net_filename , model_filename)

	imx = Image.open( img_filename )
	draw = ImageDraw.Draw(imx)
	ttfront = ImageFont.truetype('simsun.ttc', 32)
	draw.text((10, 10), car_style_rear[pre], fill=(255,0,0), font=ttfront)
	imx.show(imx)


	#show_result(info) 

if __name__ == '__main__':  
	#reload(sys)  
	#sys.setdefaultencoding('utf-8')  
	
	root = Tk()  
	root.title("Car Label Recognition")  
	root.geometry("400x200+450+210") #width x height; start coor
	
	frame = Frame(root)
	frame.pack()


	frm_2 = Frame(frame)
	Label(frm_2,text="").pack()
	var_net = StringVar()
	Entry(frm_2,textvariable=var_net,bd=1, width=25).pack(expand=1, side=LEFT) 
	Button(frm_2,text="select net",command=get_net_file,height=1,width=10).pack(side=RIGHT)
	frm_2.pack()


	frm_3 = Frame(frame)
	var_model = StringVar()  
	Entry(frm_3,textvariable=var_model,bd=1, width=25).pack(expand=1, side=LEFT) 
	Button(frm_3,text="select model",command=get_model_file,height=1,width=10).pack(side=RIGHT)
	frm_3.pack()


	frm_1 = Frame(frame)
	var_img = StringVar()  
	#Label(frm_1,text="").pack()
	Entry(frm_1,textvariable=var_img,bd=1, width=25).pack(expand=1, side=LEFT) 
	Button(frm_1,text="select image",command=get_img_file,height=1,width=10).pack(side=RIGHT)
	frm_1.pack()


	
	frm_res = Frame(root)  
	Label(frm_res,text="").pack() 
	#Label(frm_res,text="Label : ").pack(side=LEFT) 
	var_label = StringVar()
	Label(frm_res,textvariable=var_label).pack()  
	frm_res.pack()  #side=BOTTOM


	frm_D = Frame(root)  
	Button(frm_D,text="run",command=run,height=2,width=10).pack(side=LEFT)
	Button(frm_D,text="exit",command=root.quit,height=2,width=10,bg="#B0D060").pack()  
	frm_D.pack(side=BOTTOM) 
	
	
	root.mainloop()