import tensorflow as tf
from cnn import VGG16
import os, sys
import numpy as np
from config import FLAGS

current_dir = os.path.abspath('../')
sys.path.append(current_dir)
from data_helper import gen_train_or_test_batch, load_data_cifar10

vgg16 = VGG16(keep_prob=1)
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_save_path))
    x_train, y_train= load_data_cifar10(current_dir+'/data/cifar-10-batches-py/',test=True)
    correct_sum = 0
    for i in range(100):
        batch_x, batch_y = gen_train_or_test_batch(x_train, y_train, i, 100)
        feed_dic = {vgg16.is_training: False, vgg16.input_x: batch_x, vgg16.input_y: batch_y}
        prediction = sess.run(vgg16.predictions, feed_dict=feed_dic)
        correct_sum += np.sum(prediction * 1)

    print('Acc on test dataset:', (correct_sum / 10000))
