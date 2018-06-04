import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image, ImageDraw

from annotations.keypoints import HeatMap
HeatMap.dims = [12, 12]

def keypoint_index(features, start_idx):
    idx_bounds = (start_idx + np.arange(0, 2)) * HeatMap.dims[0] * HeatMap.dims[1]
    kp_indexes = np.arange(*idx_bounds)
    kp_features = features[kp_indexes]
    kp_features = kp_features.reshape(HeatMap.dims)
    kp_idx = np.unravel_index(
        np.argmax(kp_features, axis=None), kp_features.shape)
    kp_value = kp_features[kp_idx]

    kp_idx = np.array(kp_idx, dtype=np.int)
    # Change the order: (i,j) -> (j,i) [i.e., (x,y)]
    kp_idx = np.flip(kp_idx, axis=0)

    return (kp_idx, kp_value)

def predict_keypoints(kps_features, keypoints):
    def gen_kps_predictions():
        for start_idx, norm_coords in \
            zip(keypoints.indexes, keypoints.normalized_coords()):

            kp_idx, kp_value = keypoint_index(kps_features, start_idx)
            prediction = kp_value >= 0 and np.all(kp_idx == norm_coords)
            yield prediction

    return np.stack(gen_kps_predictions())

def predict_annotations(net, dataset, transformer, annotations):
    count = 0
    for imgpath, annot_data in annotations.items():
        image = Image.open(imgpath)

        for kps in annot_data:
            roi = image.crop(kps.bbox)
            roi_bbox = [0, 0, roi.size[0] - 1, roi.size[1] - 1]
            transformed_roi = transformer.preprocess('data', roi, bbox=roi_bbox)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[count] = transformed_roi
            count += 1

    ### perform classification
    output = net.forward()

    annot_predictions = np.empty(0, dtype=np.bool)
    count = 0
    for annot_data in annotations.values():
        for keypoints in annot_data:
            kps_features = net.blobs['flatten6'].data[count]
            kps_predictions = predict_keypoints(kps_features, keypoints)
            annot_predictions = np.append(annot_predictions, kps_predictions)
            count += 1

    return annot_predictions

def align_bbox_indexes(bbox_indexes):
    y = bbox_indexes.flatten()
    # The least-square solution represents vector [xmin, ymin, xmax, ymax].T
    A = np.vstack([[1, 0, 0, 0], [0, 1, 0, 0],
                   [1, 0, 0, 0], [0, 0, 0, 1],
                   [0, 0, 1, 0], [0, 1, 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    bbox = np.linalg.lstsq(A, y, rcond=None)[0]

    center_x = 0.5 * (bbox[0] + bbox[2])
    center_y = 0.5 * (bbox[1] + bbox[3])
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])
    bbox[0] = center_x - 0.5 * width
    bbox[1] = center_y - 0.5 * height
    bbox[2] = center_x + 0.5 * width
    bbox[3] = center_y + 0.5 * height

    indexes = np.array([[0, 1], [0, 3], [2, 1], [2, 3]])
    return bbox[indexes]

def indexes_to_coords(bbox_indexes, bbox):
    delta_x = float(bbox[2] - bbox[0] + 1) / float(HeatMap.dims[0])
    delta_y = float(bbox[3] - bbox[1] + 1) / float(HeatMap.dims[1])
    kps_coords = np.apply_along_axis(
        lambda xy: (xy + 0.5) * [delta_x, delta_y] + bbox[0:2], 1, bbox_indexes)
    return kps_coords

def draw_keypoints(draw, kps_coords, fill):
    points = [0, 1, 3, 2, 0]
    draw.line(list(kps_coords[points].flatten()), fill=fill, width=5)

def draw_windshield(start_indexes, kps_features, bbox, draw, fill):
    windshield_indexes = np.empty((0, 2), dtype=np.int)
    for start_idx in start_indexes:
        bbox_indexes = []
        bbox_values = []
        for idx in range(4 * start_idx, 4 * (start_idx + 1)):
            kp_idx, kp_value = keypoint_index(kps_features, idx)
            if kp_value >= 0:
                kp_idx = np.expand_dims(kp_idx, axis=0)
                bbox_indexes.append(kp_idx)
                kp_value = np.expand_dims(kp_value, axis=0)
                bbox_values.append(kp_value)

        if len(bbox_indexes) == 0:
            continue
        bbox_indexes = np.concatenate(bbox_indexes)
        bbox_values = np.concatenate(bbox_values)

        kp_idx = bbox_indexes[bbox_values.argmax()]
        kp_idx = np.expand_dims(kp_idx, axis=0)
        windshield_indexes = np.append(windshield_indexes, kp_idx, axis=0)

    if windshield_indexes.shape[0] == 4:
        kps_coords = indexes_to_coords(windshield_indexes, bbox)
        draw_keypoints(draw, kps_coords, fill)

def draw_keypoints_bboxes(start_indexes, kps_features, bbox, draw, fill):
    for start_idx in start_indexes:
        bbox_indexes = []
        for idx in range(4 * start_idx, 4 * (start_idx + 1)):
            kp_idx, kp_value = keypoint_index(kps_features, idx)
            if kp_value >= 0:
                kp_idx = np.expand_dims(kp_idx, axis=0)
                bbox_indexes.append(kp_idx)

        if len(bbox_indexes) < 4:
            continue
        bbox_indexes = np.concatenate(bbox_indexes)
        bbox_indexes = align_bbox_indexes(bbox_indexes)
        bbox_coords = indexes_to_coords(bbox_indexes, bbox)
        draw_keypoints(draw, bbox_coords, fill)

def draw_bbox(bbox, draw):
    cmap = plt.cm.Set1
    points = [0, 1, 2, 1, 2, 3, 0, 3, 0, 1]
    draw.line(list(bbox[points]), fill=mcolors.to_hex(cmap(2)), width=5)

def draw_all_keypoints(dataset, kps_features, bbox, draw):
    cmap = plt.cm.Set1

    part_names = ['stLeftFrontTyre', 'stLeftRearTyre',
                  'stRightFrontTyre', 'stRightRearTyre']
    start_indexes = [dataset.parts.index(part) for part in part_names]
    draw_keypoints_bboxes(
        start_indexes, kps_features, bbox, draw, fill=mcolors.to_hex(cmap(6)))

    part_names = ['stLeftFrontFoglamp', 'stRightFrontFoglamp',
                  'stLeftRearFoglamp', 'stRightRearFoglamp']
    start_indexes = [dataset.parts.index(part) for part in part_names]
    draw_keypoints_bboxes(
        start_indexes, kps_features, bbox, draw, fill=mcolors.to_hex(cmap(4)))

    part_names = ['stLeftFrontLight', 'stLeftRearLight',
                  'stRightFrontLight', 'stRightRearLight']
    start_indexes = [dataset.parts.index(part) for part in part_names]
    draw_keypoints_bboxes(
        start_indexes, kps_features, bbox, draw, fill=mcolors.to_hex(cmap(5)))

    part_names = ['stSunroof']
    start_indexes = [dataset.parts.index(part) for part in part_names]
    draw_keypoints_bboxes(
        start_indexes, kps_features, bbox, draw, fill=mcolors.to_hex(cmap(3)))

    part_names = ['stLeftRearView', 'stRightRearView']
    start_indexes = [dataset.parts.index(part) for part in part_names]
    draw_keypoints_bboxes(
        start_indexes, kps_features, bbox, draw, fill=mcolors.to_hex(cmap(0)))

    part_names = ['stFrontWindshieldLeftUp', 'stFrontWindshieldRightUp',
                  'stFrontWindshieldLeftBottom', 'stFrontWindshieldRightBottom']
    start_indexes = [dataset.parts.index(part) for part in part_names]
    draw_windshield(
        start_indexes, kps_features, bbox, draw, fill=mcolors.to_hex(cmap(7)))

    part_names = ['stRearWindshieldLeftUp', 'stRearWindshieldRightUp',
                  'stRearWindshieldLeftBottom', 'stRearWindshieldRightBottom']
    start_indexes = [dataset.parts.index(part) for part in part_names]
    draw_windshield(
        start_indexes, kps_features, bbox, draw, fill=mcolors.to_hex(cmap(7)))

    part_names = ['stIntakegate']
    start_indexes = [dataset.parts.index(part) for part in part_names]
    draw_keypoints_bboxes(
        start_indexes, kps_features, bbox, draw, fill=mcolors.to_hex(cmap(1)))

    part_names = ['stRearLeftLogo', 'stRearRightLogo', 'stLogo']
    start_indexes = [dataset.parts.index(part) for part in part_names]
    draw_keypoints_bboxes(
        start_indexes, kps_features, bbox, draw, fill=mcolors.to_hex(cmap(2)))
