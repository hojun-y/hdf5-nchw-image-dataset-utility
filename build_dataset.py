import os
import PIL.Image as Image
import numpy as np
import argparse
import h5py
import processing.preprocessor as preprocessor
import processing.compressor as compressor
import processing.flags as strs

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('output_path', type=str, help="Dataset output path")
parser.add_argument('images_path', type=str, help="Image data folder path")
parser.add_argument('resolution', type=int, help="Image data resolution (in pixel)")
parser.add_argument('--compress', type=str, default='blosclz',
                    choices=['blosclz', 'lz4', 'zlib', 'zstd', 'snappy', 'lz4hc'],
                    help='Image data compression method')
parser.add_argument('--crop', type=str, default='center_crop',
                    choices=['center_crop', 'random_crop', 'resize_only', 'pad_resize'],
                    help='Image cropping method')
parser.add_argument('--resize', type=str, default='bilinear',
                    choices=['bilinear', 'nearest'],
                    help='Image resizing method')
parser.add_argument('--comp_level', type=int, default=4, help='Image compression level 0-9.')
parser.add_argument('--iter', type=int, default=1, help='Processing iteration for random cropping method')
parser.add_argument('--unsupervised', action='store_true', default=False)
parser.add_argument('--grayscale', action='store_true', default=False)
args = parser.parse_args()

if os.path.exists(args.output_path):
    while True:
        print('Caution!! The HDF5 dataset already exists!\n',
              'Path: ' + args.output_path)
        get = input('Overwrite it? (y/n)')
        get = get.lower()
        if get == 'n':
            exit(1)
        elif get == 'y':
            break
        else:
            pass

channels = 3 if not args.grayscale else 1


# Path searching generator
def _search_files_generator():
    path = args.images_path
    for folder_path, _, files in os.walk(path):
        for file in files:
            str_label = folder_path[len(path):].split(os.sep)[-1] if not args.unsupervised else None
            yield [os.path.join(folder_path, file), str_label]


# Initialize compressor and preprocessor
preprocess = preprocessor.Preprocessor(args.resolution, args.resize, args.crop, channels)
compress = compressor.Compressor(args.compress, np.clip(args.comp_level, 0, 9))
print('TF session and compressor initialized.')

idx = 0
with h5py.File(args.output_path, 'w') as f:
    f.create_group(strs.DATASET_DATA_FLAG)
    f.create_group(strs.DATASET_INFO_FLAG)
    if not args.unsupervised:
        f.create_group(strs.DATASET_LABEL_FLAG)

    image_generator = _search_files_generator()
    for image, label in image_generator:
        img = Image.open(image)

        if channels == 1:
            img = img.convert('L')
            img = np.array(img)
            img = np.reshape(img, [img.shape[0], img.shape[1], 1])

        else:
            img = np.array(img)

        for i in range(args.iter):
            img_processed = preprocess.process_image(img)
            img_compressed = compress.compress_bytes(img_processed.tostring())
            f.create_dataset(strs.DATASET_DATA_FLAG + '/' + str(idx), data=np.void(img_compressed))
            if not args.unsupervised:
                f.create_dataset(strs.DATASET_LABEL_FLAG + '/' + str(idx), data=int(label))

            idx += 1

            if idx % 1000 == 0:
                print(str(idx).zfill(9) + ' Image Processed.')

    # add infos
    f.create_dataset(strs.INFO_LABELLING, data=not args.unsupervised)
    f.create_dataset(strs.INFO_MAX_INDEX, data=(idx-1))
    f.create_dataset(strs.INFO_RESOLUTION, data=args.resolution)
    f.create_dataset(strs.INFO_CHANNEL, data=channels)
    f.create_dataset(strs.INFO_VERSION, data=strs.DATASET_VERSION)
print('Dataset created. Total images: ' + str(idx))
