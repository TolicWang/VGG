import tensorflow as tf
from config import FLAGS


def conv2d_op(input, ksizes, op_num):
    with tf.name_scope('conv-%s' % op_num):
        weights = tf.Variable(tf.truncated_normal(shape=ksizes, stddev=0.1, dtype=tf.float32), name='Weights')
        bias = tf.constant(value=0, dtype=tf.float32, shape=[ksizes[3]])
        conv = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        convs = tf.nn.bias_add(conv, bias)
        return tf.nn.relu(convs)


def max_pool_op(input, strides, op_num):
    with tf.name_scope('maxpool-%s' % op_num):
        return tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=strides, padding='SAME')


def full_collection_op(input, nodes_in, nodes_out, op_num, relu=True):
    with tf.name_scope('fc-%s' % op_num):
        weights = tf.Variable(tf.truncated_normal(shape=[nodes_in, nodes_out], stddev=0.1, dtype=tf.float32))
        bias = tf.constant(value=0, dtype=tf.float32, shape=[nodes_out])
        tf.add_to_collection('l2_loss', tf.nn.l2_loss(weights))
        result = tf.nn.xw_plus_b(input, weights, bias)
        if relu:
            return tf.nn.relu(result)
        else:
            return result


class VGG16(object):
    def __init__(self,
                 ):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None,
                                                               FLAGS.input_size,
                                                               FLAGS.input_size,
                                                               FLAGS.channel], name='input-x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input-y')

        layer1 = conv2d_op(self.input_x, ksizes=FLAGS.conv1_ksizes, op_num=1)
        layer2 = conv2d_op(layer1, ksizes=FLAGS.conv2_ksizes, op_num=2)
        maxpool1 = max_pool_op(layer2, strides=FLAGS.max_poo11_strides, op_num=1)
        layer3 = conv2d_op(maxpool1, ksizes=FLAGS.conv3_ksizes, op_num=3)
        layer4 = conv2d_op(layer3, ksizes=FLAGS.conv4_ksizes, op_num=4)
        maxpool2 = max_pool_op(layer4, strides=FLAGS.max_poo12_strides, op_num=2)
        layer5 = conv2d_op(maxpool2, ksizes=FLAGS.conv5_ksizes, op_num=5)
        layer6 = conv2d_op(layer5, ksizes=FLAGS.conv6_ksizes, op_num=6)
        layer7 = conv2d_op(layer6, ksizes=FLAGS.conv7_ksizes, op_num=7)
        maxpool3 = max_pool_op(layer7, strides=FLAGS.max_poo13_strides, op_num=3)
        layer8 = conv2d_op(maxpool3, ksizes=FLAGS.conv8_ksizes, op_num=8)
        layer9 = conv2d_op(layer8, ksizes=FLAGS.conv9_ksizes, op_num=9)
        layer10 = conv2d_op(layer9, ksizes=FLAGS.conv10_ksizes, op_num=10)
        maxpool4 = max_pool_op(layer10, strides=FLAGS.max_poo14_strides, op_num=4)
        layer11 = conv2d_op(maxpool4, ksizes=FLAGS.conv11_ksizes, op_num=11)
        layer12 = conv2d_op(layer11, ksizes=FLAGS.conv12_ksizes, op_num=12)
        layer13 = conv2d_op(layer12, ksizes=FLAGS.conv13_ksizes, op_num=13)
        maxpool5 = max_pool_op(layer13, strides=FLAGS.max_poo15_strides, op_num=5)
        pool_shape = maxpool5.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(tensor=maxpool5, shape=[-1, nodes])

        layer14 = full_collection_op(input=reshaped, nodes_in=nodes, nodes_out=FLAGS.fc1_size, op_num=1)
        drop_layer14 = tf.nn.dropout(layer14, 0.5)
        layer15 = full_collection_op(drop_layer14, nodes_in=FLAGS.fc1_size, nodes_out=FLAGS.fc2_size, op_num=2)
        drop_layer15 = tf.nn.dropout(layer15, 0.5)
        layer16 = full_collection_op(drop_layer15, nodes_in=FLAGS.fc2_size, nodes_out=FLAGS.fc3_size, relu=False,
                                     op_num=3)

        self.logits = layer16

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            l2_losses = tf.add_n(tf.get_collection('l2_loss'))
            self.loss = tf.reduce_mean(cross_entropy) + l2_losses * FLAGS.l2_regul_rate

        with tf.name_scope('pre-acc'):
            self.predictions = tf.argmax(self.logits, axis=1, name='prediction')
            correct_predictions = tf.equal(self.predictions,
                                           self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


if __name__ == '__main__':
    print('\n----------------Parameters--------------')  # 在网络训练之前，先打印出来看看
    for attr, value in sorted(FLAGS.__flags.items()):
        print('{}={}'.format(attr.upper(), value))
    a = tf.random_normal(shape=[5, 224, 224, 3])
    b = conv2d_op(a, ksizes=[3, 3, 3, 64], op_num=3)
    print(b)