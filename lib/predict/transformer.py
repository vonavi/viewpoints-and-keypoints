import numpy as np
from PIL import Image

class Transformer(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.channel_swap = {}
        self.mirror = {}
        self.crop_mode = {}
        self.context_pad = {}
        self.mean_values = {}
        self.scale = {}
        self.transpose = {}

    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception('{} is not one of the net inputs: {}'.format(
                in_, self.inputs))

    def preprocess(self, in_, data, bbox):
        self.__check_input(in_)
        channel_swap = self.channel_swap.get(in_)
        mirror = self.mirror.get(in_) or False
        crop_size = self.inputs.get(in_)[2]
        crop_mode = self.crop_mode.get(in_) or 'warp'
        context_pad = self.context_pad.get(in_) or 0
        mean_values = self.mean_values.get(in_)
        scale = self.scale.get(in_) or 1.0
        transpose = self.transpose.get(in_)

        caffe_in = np.array(data).astype(np.uint8)
        if channel_swap is not None:
            caffe_in = caffe_in[:, :, channel_swap]
        caffe_in = self.__window_data(
            caffe_in, bbox, mirror, crop_size, crop_mode,
            context_pad, mean_values, scale)
        if transpose is not None:
            caffe_in = caffe_in.transpose(transpose)
        return caffe_in

    def set_channel_swap(self, in_, channel_swap):
        self.__check_input(in_)
        self.channel_swap[in_] = channel_swap

    def set_mirror(self, in_, mirror):
        self.__check_input(in_)
        self.mirror[in_] = mirror

    def set_crop_mode(self, in_, crop_mode):
        self.__check_input(in_)
        self.crop_mode[in_] = crop_mode

    def set_context_pad(self, in_, context_pad):
        self.__check_input(in_)
        self.context_pad[in_] = context_pad

    def set_mean_values(self, in_, mean_values):
        self.__check_input(in_)
        self.mean_values[in_] = mean_values

    def set_scale(self, in_, scale):
        self.__check_input(in_)
        self.scale[in_] = scale

    def set_transpose(self, in_, transpose):
        self.__check_input(in_)
        self.transpose[in_] = transpose

    @staticmethod
    def __window_data(im, bbox, mirror, crop_size, crop_mode, context_pad,
                      mean_values, scale):
        # N.B. this should be as identical as possible to the cropping
        # implementation in Caffe's WindowDataLayer, which is used while
        # fine-tuning.
        use_square = crop_mode == 'square'

        # crop window out of image and warp it
        x1, y1, x2, y2 = bbox

        pad_w = 0
        pad_h = 0
        if context_pad > 0 or use_square:
            # scale factor by which to expand the original region
            # such that after warping the expanded region to crop_size x crop_size
            # there's exactly context_pad amount of padding on each side
            context_scale = float(crop_size) / float(crop_size - 2 * context_pad)

            # compute the expanded region
            half_height = float(y2 - y1 + 1) / float(2)
            half_width = float(x2 - x1 + 1) / float(2)
            center_x = x1 + half_width
            center_y = y1 + half_height
            if use_square:
                if half_height > half_width:
                    half_width = half_height
                else:
                    half_height = half_width

            x1 = int(round(center_x - half_width * context_scale))
            x2 = int(round(center_x + half_width * context_scale))
            y1 = int(round(center_y - half_height * context_scale))
            y2 = int(round(center_y + half_height * context_scale))

            # the expanded region may go outside of the image
            # so we compute the clipped (expanded) region and keep track of
            # the extent beyond the image
            unclipped_height = y2 - y1 + 1
            unclipped_width = x2 - x1 + 1
            pad_x1 = max(0, -x1)
            pad_y1 = max(0, -y1)
            pad_x2 = max(0, x2 - im.shape[1] + 1)
            pad_y2 = max(0, y2 - im.shape[0] + 1)
            # clip bounds
            x1 = x1 + pad_x1
            x2 = x2 - pad_x2
            y1 = y1 + pad_y1
            y2 = y2 - pad_y2

            clipped_height = y2 - y1 + 1
            clipped_width = x2 - x1 + 1

            # scale factors that would be used to warp the unclipped
            # expanded region
            scale_x = float(crop_size) / float(unclipped_width)
            scale_y = float(crop_size) / float(unclipped_height)

            # size to warp the clipped expanded region to
            crop_width = int(round(clipped_width * scale_x))
            crop_height = int(round(clipped_height * scale_y))
            pad_x1 = int(round(pad_x1 * scale_x))
            pad_x2 = int(round(pad_x2 * scale_x))
            pad_y1 = int(round(pad_y1 * scale_y))
            pad_y2 = int(round(pad_y2 * scale_y))

            pad_h = pad_y1
            # if we're mirroring, we mirror the padding too (to be pedantic)
            if mirror:
                pad_w = pad_x2
            else:
                pad_w = pad_x1

            # ensure that the warped, clipped region plus the padding fits in the
            # crop_size x crop_size image (it might not due to rounding)
            if pad_h + crop_height > crop_size:
                crop_height = crop_size - pad_h
            if pad_w + crop_width > crop_size:
                crop_width = crop_size - pad_w

        roi = Image.fromarray(im).crop((x1, y1, x2 - x1 + 1, y2 - y1 + 1))
        roi = roi.resize((crop_width, crop_height), resample=Image.BICUBIC)
        if mirror:
            roi = roi.transpose(Image.FLIP_LEFT_RIGHT)
        roi = np.array(roi).astype(np.float)

        window = np.zeros((crop_size, crop_size, 3), dtype=np.float)
        if mean_values is not None:
            for i in np.arange(3):
                window[pad_h:(pad_h + crop_height), pad_w:(pad_w + crop_width), i] \
                    = (roi[:, :, i] - mean_values[i]) * scale
        else:
            window[pad_h:(pad_h + crop_height), pad_w:(pad_w + crop_width)] \
                = roi * scale
        return window
