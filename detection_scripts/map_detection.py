import numpy as np


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def func(det_filname, gt_filename, ovthresh=0.5):


    det_filname = det_filname


    fram_data = {}
    difficult = []

    with open(gt_filename, 'r') as f:

        gt_raw_data = f.readlines()
        gt_raw_data = [l.strip().split(' ') for l in gt_raw_data]
        
        for ll in gt_raw_data:
            
            if ll[2] in ['Car', 'Van']:

                difficult.append([0])
                try:
                    fram_data[int(ll[0])].append(ll)
                except:
                    fram_data[int(ll[0])] = [ll]


            # elif ll[2] == 'DontCare' :

            #     difficult.append([1])
            #     try:
            #         neg_data[imgs[int(ll[0])]].append(ll)
            #     except:
            #         neg_data[imgs[int(ll[0])]] = [ll]


    # extract gt objects for this class
    class_recs = {}
    npos = 0

    if len(fram_data.keys()) == 0:
        return 0,0,0

    for k in fram_data.keys():
        
        R = fram_data[k]
        bbox = np.array([ [float(x[6]), float(x[7]), float(x[8]), float(x[9])] for x in R])
        # difficult = np.array([0] for x in R).astype(np.bool)
        det = [False] * len(R)
        #npos = npos + sum(~difficult)
        npos += len(R)

        class_recs[k] = {'bbox': bbox, 'det': det}


    # read dets
    detfile = det_filname
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [int(x[0]) for x in splitlines]
    confidence = np.array([float(x[-1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[1:-1]] for x in splitlines])



    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]



    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):

        try:
            R = class_recs[image_ids[d]]
        except:
            fp[d] = 1
            continue

        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.

        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap




if __name__ == '__main__':


    import glob
    gt_filenames = glob.glob('D:/XiChen/KITTI/training/label_02/*.txt')
    det_filnames = glob.glob('D:/WenyuLv/workspace/py-faster-rcnn-master/test/kitti_result/vgg/train/*.txt')


    aps = []
    for gt,det in zip(gt_filenames, det_filnames):

        rec, prec, ap = func(det, gt)
        aps.append(ap)

        print ap


    print ''
    print sum(aps)/(len(aps)-1)