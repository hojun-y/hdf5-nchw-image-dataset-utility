import tensorflow as tf
import numpy as np
import numba

crop_resizemethod = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                     'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR}


class Preprocessor:
    def __init__(self, resolution, resize_method, crop_method):
        self.img = tf.placeholder(tf.float32, [None, None, 3])
        self.min_size = tf.placeholder(tf.int32, None)

        self.source = tf.divide(self.img, 255)
        self.source = tf.subtract(tf.multiply(self.source, 2), 1)  # Fit for tanh
        if crop_method == 'center_crop':
            self.process = tf.image.resize_image_with_crop_or_pad(self.source, self.min_size, self.min_size)
        if crop_method == 'resize_only':
            self.process = self.source
        if crop_method == 'pad_resize':
            self.process = \
                tf.image.resize_image_with_pad(self.source, resolution, resolution,
                                               method=crop_resizemethod[resize_method])
        if crop_method == 'random_crop':
            self.process = tf.image.random_crop(self.source, (self.min_size, self.min_size, 3))

        if crop_method != 'pad_resize':
            self.process = \
               tf.image.resize_images(self.process, (resolution, resolution), method=crop_resizemethod[resize_method])
        self.final = tf.transpose(self.process, [2, 0, 1])  # HWC to CHW

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    @numba.jit
    def _get_min_size(self, img):
        return np.min(np.shape(img)[:2])

    def process_image(self, img):
        min_size = self._get_min_size(img)
        img = self.sess.run(self.final, feed_dict={self.img: img, self.min_size: min_size})
        return img
