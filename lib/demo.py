import os
import sys
caffe_root = os.path.join('D:', os.sep, 'Repos', 'caffe')
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

import collections
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw
from datasets.veh_keypoints import VehKeypoints
from predict.transformer import Transformer
from predict.veh_keypoints import draw_bbox, draw_all_keypoints

LIB_PATH = os.path.dirname(os.path.realpath(__file__))
CACHE_PATH = os.path.join(LIB_PATH, '..', 'cachedir')
VEH_KEYPOINTS_PATH = os.path.join(LIB_PATH, '..', 'data', 'veh_keypoints')
net_proto = os.path.join(
    LIB_PATH, '..', 'prototxts', 'vgg_veh_conv12', 'deploy.prototxt')
net_weights = os.path.join(
    CACHE_PATH, 'snapshots', 'vgg_veh_conv12_iter_70000.caffemodel')

imageset_path = os.path.join('D:', os.sep, 'Data', 'key_point_task', '20171211')
annotation_path = os.path.join(imageset_path, 'HuanChengRoad_014_mediacode.csv')

def predict_keypoints(net, dataset, transformer, frame_car_bboxes):
    count = 0
    for frame_num, car_bboxes in frame_car_bboxes.items():
        image_path = os.path.join(
            imageset_path, 'orig_frames',
            'output_' + format(frame_num + 1, '04d') + '.bmp')
        if not os.path.isfile(image_path):
            continue
        image = Image.open(image_path)

        for car_bbox in car_bboxes:
            roi = image.crop(car_bbox)
            roi_bbox = [0, 0, roi.size[0] - 1, roi.size[1] - 1]
            transformed_roi = transformer.preprocess('data', roi, bbox=roi_bbox)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[count] = transformed_roi
            count += 1

    ### perform classification
    output = net.forward()

    count = 0
    for frame_num, car_bboxes in frame_car_bboxes.items():
        image_path = os.path.join(
            imageset_path, 'orig_frames',
            'output_' + format(frame_num + 1, '04d') + '.bmp')
        if not os.path.isfile(image_path):
            continue
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        for car_bbox in car_bboxes:
            draw_bbox(car_bbox, draw)
            kps_features = net.blobs['flatten6'].data[count]
            draw_all_keypoints(dataset, kps_features, car_bbox, draw)
            count += 1

        output_path = os.path.join(
            imageset_path, 'marked_frames',
            'output_' + format(frame_num + 1, '04d') + '.bmp')
        image.save(output_path, 'BMP')

annotation = pd.read_csv(annotation_path)
car_id_rows = annotation['ID']
frame_num_rows = annotation['frame_num']
car_x1_rows = annotation['car_x1']
car_x2_rows = annotation['car_x2']
car_y1_rows = annotation['car_y1']
car_y2_rows = annotation['car_y2']

car_bboxes_all = collections.defaultdict(list)
for car_id, frame_num, car_x1, car_x2, car_y1, car_y2 in zip(
        car_id_rows, frame_num_rows, car_x1_rows, car_x2_rows, car_y1_rows, car_y2_rows):
    car_id = int(car_id)
    frame_num = int(frame_num)
    bbox = np.array(list(map(int, [car_x1, car_y1, car_x2, car_y2])), dtype=np.int)
    car_bboxes_all[car_id].append((frame_num, bbox))

car_interp_bboxes = collections.defaultdict(list)
for _, values in car_bboxes_all.items():
    frame_numbers, car_bboxes = zip(*values)
    frame_numbers = np.array(frame_numbers, dtype=np.int)
    car_bboxes = np.array(car_bboxes, dtype=np.int)

    for frame_num in range(min(frame_numbers), max(frame_numbers) + 1):
        car_bbox = np.empty(4, dtype=np.float)
        for i in range(4):
            car_bbox[i] = np.interp(frame_num, frame_numbers, car_bboxes[:, i])
        car_interp_bboxes[frame_num].append(car_bbox)

caffe.set_mode_gpu()
net = caffe.Net(net_proto, net_weights, caffe.TEST)
input_shape = net.blobs['data'].data.shape
batch_size = input_shape[0]

dataset = VehKeypoints(root=VEH_KEYPOINTS_PATH)

# create transformer for the input called 'data'
transformer = Transformer({'data': input_shape})
transformer.set_mirror('data', False)
transformer.set_crop_mode('data', 'warp')
transformer.set_context_pad('data', 1)
transformer.set_transpose('data', (2, 0, 1)) # move image channels to outermost dimension
transformer.set_channel_swap('data', (2, 1, 0)) # swap channels from RGB to BGR
mean_values = [102.9801, 115.9465, 122.7717] # magical numbers given by Ross
transformer.set_mean_values('data', mean_values)

part_car_bboxes = collections.defaultdict(list)
img_size = 0
img_total = len(car_interp_bboxes)
for frame_num, car_bboxes in car_interp_bboxes.items():
    if img_size + len(car_bboxes) > batch_size:
        predict_keypoints(net, dataset, transformer, part_car_bboxes)
        part_car_bboxes.clear()
        img_size = 0

    part_car_bboxes[frame_num] = car_bboxes
    img_size += len(car_bboxes)

predict_keypoints(net, dataset, transformer, part_car_bboxes)
