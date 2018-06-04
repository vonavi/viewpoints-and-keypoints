import os
import sys
caffe_root = os.path.join('D:', os.sep, 'Repos', 'caffe')
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

import collections
import numpy as np
from PIL import Image

from datasets.veh_keypoints import VehKeypoints
from preprocess.keypoints import Keypoints
from annotations.veh_keypoints import Annotations
from predict.transformer import Transformer
from predict.veh_keypoints import predict_annotations

LIB_PATH = os.path.dirname(os.path.realpath(__file__))
CACHE_PATH = os.path.join(LIB_PATH, '..', 'cachedir')
VEH_KEYPOINTS_PATH = os.path.join(LIB_PATH, '..', 'data', 'veh_keypoints')
net_proto = os.path.join(
    LIB_PATH, '..', 'prototxts', 'vgg_veh_conv12', 'deploy.prototxt')
net_weights = os.path.join(
    CACHE_PATH, 'snapshots', 'vgg_veh_conv12_iter_70000.caffemodel')

net = caffe.Net(net_proto, net_weights, caffe.TEST)
input_shape = net.blobs['data'].data.shape
caffe.set_mode_gpu()
batch_size = input_shape[0]

# create transformer for the input called 'data'
transformer = Transformer({'data': input_shape})
transformer.set_mirror('data', False)
transformer.set_crop_mode('data', 'warp')
transformer.set_context_pad('data', 1)
transformer.set_transpose('data', (2, 0, 1)) # move image channels to outermost dimension
transformer.set_channel_swap('data', (2, 1, 0)) # swap channels from RGB to BGR
mean_values = [102.9801, 115.9465, 122.7717] # magical numbers given by Ross
transformer.set_mean_values('data', mean_values)

dataset = VehKeypoints(root=VEH_KEYPOINTS_PATH)
imgset = dataset.read_set(os.path.join(CACHE_PATH, 'veh_keypoints_val.txt'))

annot_predictions = np.empty(0, dtype=np.bool)
part_annotations = collections.defaultdict(list)
img_size = 0

for imgpath in imgset:
    img_annot = Annotations(dataset=dataset, imgpath=imgpath)
    img_bbox = np.array([0, 0, img_annot.width - 1, img_annot.height - 1])
    keypoints = Keypoints(
        class_idx=0, bbox=img_bbox, coords=img_annot.coords,
        start_idx=0, kps_flips=dataset.kps_flips)

    if img_size + 1 > batch_size:
        preds = predict_annotations(net, dataset, transformer, part_annotations)
        annot_predictions = np.append(annot_predictions, preds)
        part_annotations.clear()
        img_size = 0

    part_annotations[imgpath] = [keypoints]
    img_size += 1

preds = predict_annotations(net, dataset, transformer, part_annotations)
annot_predictions = np.append(annot_predictions, preds)

correct_predictions = annot_predictions.nonzero()[0]
correct_frac = float(correct_predictions.size) / float(annot_predictions.size)
print('Correct predictions: {}'.format(correct_frac))
