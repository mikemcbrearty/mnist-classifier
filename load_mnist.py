import numpy as np
import os

cwd = os.getcwd()
mnist_dir = './mnist_archive'

# MNIST data from http://yann.lecun.com/exdb/mnist/
train_images_file = 'train-images.idx3-ubyte'
train_labels_file = 'train-labels.idx1-ubyte'
test_images_file = 't10k-images.idx3-ubyte'
test_labels_file = 't10k-labels.idx1-ubyte'
images_magic_num = 2051
labels_magic_num = 2049

def _read_images(path):
    with open(path, 'rb') as f:
        magic_num = int.from_bytes(f.read(4), byteorder='big')
        assert(magic_num == images_magic_num)
        num_images = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        columns = int.from_bytes(f.read(4), byteorder='big')

        data = f.read()

        num_pixels = rows*columns
        images = np.zeros((num_images, num_pixels), dtype=np.float64)

        n = 0
        while n < num_images:
            i = 0    
            while i < num_pixels:
                images[n][i] = data[n*num_pixels + i]
                i += 1
            n += 1
        return images


def read_train_images():
    train_images_path = os.path.join(cwd, mnist_dir, train_images_file)
    return _read_images(train_images_path)


def read_test_images():
    test_images_path = os.path.join(cwd, mnist_dir, test_images_file)
    return _read_images(test_images_path)


def _read_labels(path):
    labels = []
    with open(path, 'rb') as f:
        magic_num = int.from_bytes(f.read(4), byteorder='big')
        assert(magic_num == labels_magic_num)
        num_items = int.from_bytes(f.read(4), byteorder='big')
        labels = np.zeros((num_items,), dtype=np.int64)
        data = f.read()
        i = 0
        while i < num_items:
            labels[i] = data[i]
            i += 1
        return labels


def read_train_labels():
    train_labels_path = os.path.join(cwd, mnist_dir, train_labels_file)
    return _read_labels(train_labels_path)


def read_test_labels():
    test_labels_path = os.path.join(cwd, mnist_dir, test_labels_file)
    return _read_labels(test_labels_path)


def _print_image(image):
    row_len = 28
    col_len = 28
    for r in range(row_len):
        row = image[r*row_len : (r+1)*row_len]
        row = [f"{px:3}" for px in row]
        print(f"{r:2}: {''.join(row)}")

