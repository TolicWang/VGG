import os
import sys
import numpy as np

current_dir = '../'
sys.path.append(current_dir)


def load_data_cifar(file=current_dir + '/data/cifar-100-python/train'):
    import pickle
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
    fo.close()
    x = dicts[b'data']
    y = dicts[b'fine_labels']
    y = np.reshape(y,len(y))
    filenames = dicts[b'filenames']
    shuffled_index = np.random.permutation(len(y))
    x = x[shuffled_index]
    y = y[shuffled_index]

    return x, y, filenames


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


def visualize(index=0):
    import matplotlib.pyplot as plt
    from PIL import Image
    x, y, filenames = load_data_cifar()
    index = np.random.randint(0, len(y)) if index >= len(y) else index
    print('The picture is ', filenames[index])
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
    x, y, _ = load_data_cifar()
    xx, yy = gen_train_or_test_batch(x, y, 20, 50)
    print(xx)
    # visualize(100)
