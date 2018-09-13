import tensorflow as tf
from config import FLAGS
from cnn import VGG16
import datetime
import os, sys
import numpy as np

current_dir = os.path.abspath('../')
sys.path.append(current_dir)
from data_helper import gen_train_or_test_batch, load_data_cifar

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定一个GPU
print('\n----------------Parameters--------------')  # 在网络训练之前，先打印出来看看
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('----------------Parameters--------------\n')
x, y, _ = load_data_cifar()
x_train,y_train = x[:40000],y[:40000]
x_dev,y_dev = x[-10000:],y[-10000:]


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        vgg16 = VGG16()
        global_step = tf.Variable(0, trainable=False)
        with tf.device('/gpu:0'):
            optimaizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            # train_step = tf.train.GradientDescentOptimizer(
            #     FLAGS.learning_rate).minimize(loss=vgg16.loss, global_step=global_step)
            # grad = optimaizer.compute_gradients(loss=vgg16.loss)
            train_step = optimaizer.minimize(loss=vgg16.loss, global_step=global_step)

            # train_step = tf.train.AdamOptimizer(1e-3).minimize(loss=cnn.loss, global_step=global_step)
    saver = tf.train.Saver()
    if os.path.exists(FLAGS.model_save_path + 'checkpoint'):
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_save_path))
        print('load model!\n')
    else:
        sess.run(tf.global_variables_initializer())
    last = datetime.datetime.now()
    epoches = 0
    for i in range(999999900000):

        batch_x, batch_y = gen_train_or_test_batch(x_train, y_train, i, FLAGS.batch_size)
        feed_dic = {vgg16.input_x: batch_x, vgg16.input_y: batch_y}
        _, loss, acc = sess.run([train_step, vgg16.loss, vgg16.accuracy], feed_dict=feed_dic)

        if (i % 50) == 0:
            now = datetime.datetime.now()
            print('loss:%.7f,acc on train :%.7f---time:%s'%(loss, acc, now - last))
            last = now
        if (i%1300) ==0:
            print('========> epoches:',epoches)
            epoches +=1
        if (i % FLAGS.save_freq) == 0:
            print('Iterations:  ', i)
            saver.save(sess, os.path.join(FLAGS.model_save_path,
                                          FLAGS.model_name),
                       global_step=global_step, write_meta_graph=False)
            correct_sum = 0
            for i in range(100):
                batch_x, batch_y = gen_train_or_test_batch(x_dev, y_dev, i, 100)
                feed_dic = {vgg16.input_x: batch_x, vgg16.input_y: batch_y}
                prediction = sess.run(vgg16.predictions, feed_dict=feed_dic)
                correct_sum += np.sum(prediction*1)
            print('Acc on dev dataset:', (correct_sum / 10000))
