solver = caffe.SGDSolver('./solver.prototxt')
solver.net.save('123.caffemodel')
solver.net.copy_from('123.caffemodel')



# do forward
forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
if cfg.TEST.HAS_RPN:
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
else:
    forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
blobs_out = net.forward(**forward_kwargs)



solver.step(1000)

