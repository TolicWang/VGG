import os
import sys
import numpy as np
import random
import pickle




def load_data_cifar100(file='./data/cifar-100-python/train'):
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
    fo.close()
    x = dicts[b'data']
    y = dicts[b'fine_labels']
    y = np.reshape(y, len(y))
    filenames = dicts[b'filenames']
    np.random.seed(40)
    shuffled_index = np.random.permutation(len(y))
    x = x[shuffled_index]
    y = y[shuffled_index]

    return x, y, filenames


def load_data_cifar10(file='./data/cifar-10-batches-py/',test=False):
    if test:
        dir = file+'test_batch'
        with open(dir, 'rb') as fo:
            dicts = pickle.load(fo, encoding='bytes')
        fo.close()
        x = dicts[b'data']
        y = dicts[b'labels']
    else:
        for i in range(1, 6):
            dir = file + 'data_batch_%d' % i
            with open(dir, 'rb') as fo:
                dicts = pickle.load(fo, encoding='bytes')
            fo.close()
            temp_x = dicts[b'data']
            temp_y = dicts[b'labels']
            if i == 1:
                x = temp_x
                y = temp_y
            else:
                x = np.vstack((x, temp_x))
                y.extend(temp_y)
    np.random.seed(40)
    shuffled_index = np.random.permutation(len(y))
    x = x[shuffled_index]
    y = y[shuffled_index]
    return x, y

def per_image_standardization(images):
    batch_size = images.shape[0]
    for index in range(batch_size):
        image = images[index]
        num_pixels = np.prod(image.shape, dtype=np.float32)
        image_mean = np.mean(image)
        variance = (np.mean(np.square(image)) -
                    np.square(image_mean))
        variance = variance if variance > 0 else 0
        stddev = np.sqrt(variance)
        # Apply a minimum normalization that protects us against uniform images.
        min_stddev = 1 / np.sqrt(num_pixels)
        pixel_value_scale = np.maximum(stddev, min_stddev)
        pixel_value_offset = image_mean
        image = np.subtract(image, pixel_value_offset)
        image = np.divide(image, pixel_value_scale)
        images[index] = image
    return images


def gen_train_or_test_batch(x_train, y_train, begin=0, batch_size=64):
    data_size = len(y_train)
    start = (begin * batch_size) % data_size
    end = min(start + batch_size, data_size)
    y = y_train[start:end]
    x = x_train[start:end].reshape(len(y), 3, 32, 32).astype('float')
    standardized_images = per_image_standardization(images=x)
    standardized_images_trans = standardized_images.transpose(0, 2, 3, 1)
    return standardized_images_trans, y


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def _random_flip_updown(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.flipud(batch[i])
    return batch


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_flip_updown(batch)
    # batch = _random_crop(batch, [32, 32], 4)
    return batch


def visualize(index=0):
    import matplotlib.pyplot as plt
    from PIL import Image
    # x, y, filenames = load_data_cifar100()
    x, y= load_data_cifar10()
    index = np.random.randint(0, len(y)) if index >= len(y) else index
    # print('The picture is ', filenames[index])
    image = x[index].reshape(3, 32, 32)
    # 得到RGB通道
    r = Image.fromarray(image[0]).convert('L')
    g = Image.fromarray(image[1]).convert('L')
    b = Image.fromarray(image[2]).convert('L')
    image = Image.merge("RGB", (r, g, b))
    # 显示图片
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':

    x,y=load_data_cifar10(test=True)
    # x, y, _ = load_data_cifar100()
    # xx, yy = gen_train_or_test_batch(x, y, 20, 50)
    # print(xx)
    visualize(200)
