def _crops(img, size):
    crops = np.zeros([5]+list(size)+[3])
    crops[0, ...] = img[0:size[0], 0:size[1], :].shape
    crops[1, ...] = img[0:size[0], img.shape[1]-size[1]:, :].shape
    crops[2, ...] = img[img.shape[0]-size[0]:, 0:size[1], :].shape
    crops[3, ...] = img[img.shape[0]-size[0]:, img.shape[1]-size[1]:, :].shape
    crops[4, ...] = img[(img.shape[0]-size[0])/2:(img.shape[0]-size[0])/2+size[0],\
                        (img.shape[1]-size[1])/2:(img.shape[1]-size[1])/2+size[1], :].shape
    return crops

def image_preprocess(img):
    #img = io.imread(img_path)
    img = transform.resize(img, [224,224])*255.0
    #img = transform.rescale(img, 0.3)*255.0
    #img = _crops(img, [224,224])[4]
    img = img - [104.0, 117.0, 123.0]
    img = img[:,:,[2,1,0]]
    img = np.transpose(img, [2, 0, 1])
    return np.expand_dims(img, 0)


def test_acc(iters):
    
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net('./models/ResNet-152-test.prototxt', \
                './models/outputs/resnet-152_rescale_iter_{}.caffemodel'.format(iters), \
                caffe.TEST)
    
    with open('./data/rescale_data.txt') as f: #rescale_test
        lins = f.readlines()
        
    n=0
    for item in lins:
        lin, label = item.strip().split(' ')[0],int(item.strip().split(' ')[-1])
        img = io.imread(lin)
        #pool = Pool(32)
        #imgsx = pool.map(image_preprocess, batch_img_path_list)
        #pool.close()
        #pool.join()
        net.blobs['data'].data[...] = image_preprocess(img)
        if np.argmax(net.forward()['res']) == label:
            n+=1
        else:
            pass
            
    print 'ietrs: {}, n: {}, acc: {}'.format(iters, n, n*1.0/len(lins))


if __name__ == '__main__':

    for iterx in [ 1000*i for i in range(1,10)]
        test_acc(iterx)
