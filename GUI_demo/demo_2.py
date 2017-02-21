#coding=utf-8

import sys  
import time   
import tkFileDialog  
import tkMessageBox  
from   Tkinter import *  

sys.path.insert(0, '')
import caffe

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io
from PIL import Image, ImageDraw, ImageFont
import logging

car_style_rear = [u'Honda思铂睿2009-2013',
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

part_labels = [
'Dashboard',         
'Engine',               
'Lamp_Front',           
'Lamp_Rear',          
'Outside_Front',        
'Outside_Front_Left',  
'Outside_Rear',   
'Steering_Center_Panel',
'Trunk',         
'Wheel']  

drive_labels = [u'手动', u'自动']

car_style_front = [u'Honda思铂睿2009-2010',
u'Honda思铂睿2013',
u'Honda思铂睿2015',
u'Honda雅阁2003-2007',
u'Honda雅阁2008-2010',
u'Honda雅阁2011-2013',
u'Honda雅阁2014-2015',
u'Honda雅阁1998-2001',
u'VW帕萨特2000-2005',
u'VW帕萨特2006-2008',
u'VW帕萨特2010',
u'VW帕萨特2011-2015',
u'VW迈腾2007-2010',
u'VW迈腾2011-2015' ]

#part_deploy = './model/ResNet-152-part-deploy.prototxt'
#part_model = './model/ResNet-152-part-deploy.caffemodel' 
part_deploy = './model/googlenet_part.prototxt'
part_model = './model/googlenet_part10_queue_iter_1000.caffemodel'

drive_deploy = './model/googlenet_drive.prototxt'
drive_model = './model/googlenet_drive.caffemodel'

rear_deploy = './model/ResNet-101-rear-deploy.prototxt'
rear_model = './model/ResNet-101-rear-deploy.caffemodel'

front_deploy = './model/ResNet-101-front-deploy.prototxt'
front_model = './model/ResNet-101-front-deploy.caffemodel'


img_filename = u''

part_net = ''
drive_net = ''
rear_net = ''
front_net = ''

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


def net_init():
	global part_net
	global drive_net
	global front_net
	global rear_net

	init_start = time.time()

	#caffe.set_device(0)
	#caffe.set_mode_gpu()

	part_net = caffe.Net(part_deploy, part_model, caffe.TEST)
	drive_net = caffe.Net(drive_deploy, drive_model, caffe.TEST)
	front_net = caffe.Net(front_deploy, front_model, caffe.TEST)
	rear_net = caffe.Net(rear_deploy, rear_model, caffe.TEST)

	logging.warning('\n\n------Loading models using time: {:.5f}s------\n'.format( time.time()- init_start))

def image_preprocess(img, size=[224, 224]):

	img = transform.resize(img, size)*255.0
	img = img - [123.0, 117.0, 104.0]
	img = img[:,:,[2,1,0]]
	img = np.transpose(img, [2, 0, 1])
	return np.expand_dims(img, 0)


def run():

	global part_net
	global drive_net
	global front_net
	global rear_net

	start = time.time()

	img = plt.imread(img_filename)

	#img = io.imread(data_root + lin_part[0])
	imged1 = image_preprocess(img)

	try:
		img1 = transform.rescale(img, 224.0/min(img.shape[:2])+0.005)
		off_h = max(0, (img1.shape[0] - 224)/2-1)
		off_w = max(0, (img1.shape[1] - 224)/2-1)
		img1 = img1[off_h: off_h+224, off_w: off_w+224, :]
		imged2 = image_preprocess(img1)

		imged = np.vstack([imged1, imged2])

	except:
		imged = imged1


	part_net.blobs['data'].reshape(*(imged.shape))
	part_net.blobs['data'].data[...] = imged


	res = part_net.forward()['prob']


	if res.shape[0] == 2:
		pred = np.argmax(res, 1)[np.argmax(np.max(res, 1))]
		prob = np.max(np.max(res, 1))
	else:
		pred = np.argmax(res[0])
		prob = np.max(res[0])

	info_part = part_labels[pred]
	part_prob = prob

	if pred == 6: #outside_rear
		
		rear_net.blobs['data'].reshape(*(imged.shape))
		rear_net.blobs['data'].data[...] = imged

		res = rear_net.forward()['prob']

		if res.shape[0] == 2:
			pred_rear = np.argmax(res, 1)[np.argmax(np.max(res, 1))]
			prob = np.max(np.max(res, 1))
		else:
			pred_rear = np.argmax(res[0])
			prob = np.max(res[0])

		info_detail = car_style_rear[pred_rear]

	
	elif pred == 4 : #outside_front

		front_net.blobs['data'].reshape(*(imged.shape))
		front_net.blobs['data'].data[...] = imged

		res = front_net.forward()['prob']

		if res.shape[0] == 2:
			pred_front = np.argmax(res, 1)[np.argmax(np.max(res, 1))]
			prob = np.max(np.max(res, 1))
		else:
			pred_front = np.argmax(res[0])
			prob = np.max(res[0])

		info_detail = car_style_front[pred_front]

	elif pred == 7: #steer_center_panel

		drive_net.blobs['data'].reshape(*(imged.shape))
		drive_net.blobs['data'].data[...] = imged

		res = drive_net.forward()['prob']

		if res.shape[0] == 2:
			pred_drive = np.argmax(res, 1)[np.argmax(np.max(res, 1))]
			prob = np.max(np.max(res, 1))
		else:
			pred_drive = np.argmax(res[0])
			prob = np.max(res[0])

		info_detail = drive_labels[pred_drive]

	else:

		info_detail = ''

	if info_detail:
		info = u'车部位: {}, 置信度: {:.4f};\n 详细信息: {}, 置信度: {:.4f}'.format(info_part, part_prob, info_detail, prob)
	else:
		info = u'车部位: {}, 置信度: {:.4f}\n'.format(info_part, part_prob)
	
	logging.warning('\n\n------Processing this image using time: {:.5f}s------\n'.format(time.time()-start))


	var_info.set(info)
	

	imx = Image.open( img_filename )
	draw = ImageDraw.Draw(imx)
	
	if img.shape[1] > 800:
		ttfront = ImageFont.truetype('simsun.ttc', 30)
	else:
		ttfront = ImageFont.truetype('simsun.ttc', 18)

	width = draw.textsize(info, font=ttfront)[0]
    #width2= draw.textsize(info, font=ttfront)
    #width = width1[0] if width1[0]>width2[0] else width2[0]

	im = Image.new('RGBA', (width+2*10, 50), (0, 0, 255, 180))
	r,g,b,a = im.split()

	
	imx.paste(im, (0,0), mask=a)

	draw.text((10, 10), info, fill=(255,255,255,0), font=ttfront)

	imx.show()
	
	show_result(info)

#	img = Image.open(imgs[0])
#	#im.paste('red',box=(10, 200),mask=None)
#	
#	im = Image.new('RGBA', (img.size[0], 50), (255, 255, 255, 200))
#	draw = ImageDraw.Draw(im)
#	draw.rectangle((0, 0, img.size[0], 50), fill=(227, 198, 198, 200))
#	r,g,b,a = im.split()
#	
#	ttfront = ImageFont.truetype('simsun.ttc', 24)
#	draw.text((10, 10), 'xxx', fill=(255,0,0, 0), font=ttfront)
#	img.paste(im, (0,0), mask=a)	 
#	

if __name__ == '__main__':  
	#reload(sys)
	#sys.setdefaultencoding('utf-8') 
	
	root = Tk()
	root.title("Car Label Recognition")  
	root.geometry("350x150+450+210") #width x height; start coor
	
	frame = Frame(root)
	frame.pack()


#	net_init()

#	frm_2 = Frame(frame)
#	Label(frm_2,text="").pack()
#	var_net = StringVar()
#	Entry(frm_2,textvariable=var_net,bd=1, width=25).pack(expand=1, side=LEFT) 
#	Button(frm_2,text="select net",command=get_net_file,height=1,width=10).pack(side=RIGHT)
#	frm_2.pack()
#
#
#	frm_3 = Frame(frame)
#	var_model = StringVar()  
#	Entry(frm_3,textvariable=var_model,bd=1, width=25).pack(expand=1, side=LEFT) 
#	Button(frm_3,text="select model",command=get_model_file,height=1,width=10).pack(side=RIGHT)
#	frm_3.pack()


	frm_1 = Frame(frame)
	var_img = StringVar()  
	Label(frm_1,text="").pack()
	Entry(frm_1,textvariable=var_img,bd=1, width=30).pack(expand=1, side=LEFT) 
	Button(frm_1,text="select image",command=get_img_file,height=1,width=10).pack(side=RIGHT)
	frm_1.pack()

	
	frm_res = Frame(root)  
	Label(frm_res,text="").pack() 
	#Label(frm_res,text="Label : ").pack(side=LEFT) 
	var_info = StringVar()
	Label(frm_res,textvariable=var_info).pack()  
	frm_res.pack()


	frm_D = Frame(root)  
	Button(frm_D,text="model",command=net_init,height=2,width=10).pack(side=LEFT)
	Button(frm_D,text="run",command=run,height=2,width=10).pack(side=LEFT)
	Button(frm_D,text="exit",command=root.quit,height=2,width=10,bg="#B0D060").pack()  
	frm_D.pack(side=BOTTOM) 
	
	
	root.mainloop()