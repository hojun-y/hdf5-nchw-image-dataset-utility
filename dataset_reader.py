import h5py
import blosc
import numba
import numpy as np


DATASET_DATA_FLAG = 'data'
DATASET_LABEL_FLAG = 'label'
DATASET_INFO_FLAG = 'info'

INFO_RESOLUTION = 'info/res'
INFO_MAX_INDEX = 'info/nidx'
INFO_LABELLING = 'info/label'
INFO_CHANNEL = 'info/ch'
INFO_VERSION = 'info/ver'
DATASET_VERSION = 1


class HDF5DatasetReader:
    def __init__(self, path):
        self.file = h5py.File(path, 'r')

        # Version check
        file_ver = int(np.array(self.file[INFO_VERSION]))
        if file_ver != DATASET_VERSION:
            raise RuntimeWarning("The version of the dataset does not match the "
                                 "version of this dataset reader. There is a possibility"
                                 "of an error while getting batch data. \n"
                                 "Dataset version: V" + str(file_ver))

        self._idx_max = int(np.array(self.file[INFO_MAX_INDEX]))
        self._resolution = int(np.array(self.file[INFO_RESOLUTION]))
        self._channels = int(np.array(self.file[INFO_CHANNEL]))
        self._is_labeled = bool(np.array(self.file[INFO_LABELLING]))

        self._data_generator = self._get_next_data()

    @numba.jit
    def _reshape(self, mat):
        return mat.reshape([-1, self._resolution, self._resolution])

    def _get_next_data(self):
        while True:
            for idx in range(self._idx_max):
                compressed_data = self.file[DATASET_DATA_FLAG + '/' + str(idx)]
                img_bytes = blosc.decompress(np.array(compressed_data))
                img = self._reshape(np.frombuffer(img_bytes, dtype=np.float32))
                label = int(np.array(self.file[DATASET_LABEL_FLAG + '/' + str(idx)])) if self._is_labeled else None
                yield img, label

    @numba.jit
    def get_batch(self, batch_size, label=True):
        img_batch = np.empty([batch_size, self._channels, self._resolution, self._resolution])
        label_batch = None

        if label and self._is_labeled:
            label_batch = np.empty([batch_size])

        for i in range(batch_size):
            data, label = self._data_generator.__next__()
            img_batch[i] = data
            if label and self._is_labeled:
                label_batch[i] = label
        return img_batch, label_batch
