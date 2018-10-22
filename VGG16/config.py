import tensorflow as tf

# input size
tf.flags.DEFINE_integer(flag_name='input_size', default_value=32, docstring='input_size')
tf.flags.DEFINE_integer(flag_name='channel', default_value=3, docstring='input_channel')


#   layer1 parameters
tf.flags.DEFINE_string(flag_name='conv1_ksizes', default_value=[3, 3, 3, 64], docstring='conv1_ksizes')
#   layer2 parameters
tf.flags.DEFINE_string(flag_name='conv2_ksizes', default_value=[3, 3, 64, 64], docstring='conv2_ksizes')
#   max pool
tf.flags.DEFINE_string(flag_name='max_poo11_strides', default_value=[1, 1, 1, 1], docstring='max_poo11_strides')

#   layer3 parameters
tf.flags.DEFINE_string(flag_name='conv3_ksizes', default_value=[3, 3, 64, 128], docstring='conv3_ksizes')
#   layer4 parameters
tf.flags.DEFINE_string(flag_name='conv4_ksizes', default_value=[3, 3, 128, 128], docstring='conv4_ksizes')
#   max pool
tf.flags.DEFINE_string(flag_name='max_poo12_strides', default_value=[1, 2, 2, 1], docstring='max_poo12_strides')

#   layer5 parameters
tf.flags.DEFINE_string(flag_name='conv5_ksizes', default_value=[3, 3, 128, 256], docstring='conv5_ksizes')
#   layer6 parameters
tf.flags.DEFINE_string(flag_name='conv6_ksizes', default_value=[3, 3, 256, 256], docstring='conv6_ksizes')
#   layer7 parameters
tf.flags.DEFINE_string(flag_name='conv7_ksizes', default_value=[3, 3, 256, 256], docstring='conv7_ksizes')
#   max pool
tf.flags.DEFINE_string(flag_name='max_poo13_strides', default_value=[1, 2, 2, 1], docstring='max_poo13_strides')

#   layer8 parameters
tf.flags.DEFINE_string(flag_name='conv8_ksizes', default_value=[3, 3, 256, 512], docstring='conv8_ksizes')
#   layer9 parameters
tf.flags.DEFINE_string(flag_name='conv9_ksizes', default_value=[3, 3, 512, 512], docstring='conv9_ksizes')
#   layer10 parameters
tf.flags.DEFINE_string(flag_name='conv10_ksizes', default_value=[3, 3, 512, 512], docstring='conv10_ksizes')
#   max pool
tf.flags.DEFINE_string(flag_name='max_poo14_strides', default_value=[1, 2, 2, 1], docstring='max_poo14_strides')

#   layer11 parameters
tf.flags.DEFINE_string(flag_name='conv11_ksizes', default_value=[3, 3, 512, 512], docstring='conv11_ksizes')
#   layer12 parameters
tf.flags.DEFINE_string(flag_name='conv12_ksizes', default_value=[3, 3, 512, 512], docstring='conv12_ksizes')
#   layer13 parameters
tf.flags.DEFINE_string(flag_name='conv13_ksizes', default_value=[3, 3, 512, 512], docstring='conv13_ksizes')
#   max pool
tf.flags.DEFINE_string(flag_name='max_poo15_strides', default_value=[1, 2, 2, 1], docstring='max_poo15_strides')

#   layer14 parameters
tf.flags.DEFINE_integer(flag_name='fc1_size', default_value=4096, docstring='fc1_size')
#   layer15 parameters
tf.flags.DEFINE_integer(flag_name='fc2_size', default_value=4096, docstring='fc2_size')
#   layer16 parameters
tf.flags.DEFINE_integer(flag_name='fc3_size', default_value=10, docstring='fc3_size')


#   training parameters
tf.flags.DEFINE_integer(flag_name='training_ite', default_value=800000, docstring='training_ite')
tf.flags.DEFINE_boolean(flag_name='allow_soft_placement', default_value='True',
                        docstring='allow_soft_placement')  # 找不到指定设备时，是否自动分配
tf.flags.DEFINE_float(flag_name='learning_rate', default_value=0.8, docstring='learning_rate')
tf.flags.DEFINE_integer(flag_name='batch_size', default_value=32, docstring='batch_size')
tf.flags.DEFINE_float(flag_name='l2_regul_rate', default_value=0.0001, docstring='l2_regul_rate')
tf.flags.DEFINE_float(flag_name='dropout_keep_pro', default_value=0.5, docstring='drop_keep_pro')

tf.flags.DEFINE_string(flag_name='model_save_path', default_value='./model/', docstring='model_save_path')
tf.flags.DEFINE_string(flag_name='model_name', default_value='model.ckpt', docstring='model_name')
tf.flags.DEFINE_integer(flag_name='save_freq', default_value=5000, docstring='save_freq')
tf.flags.DEFINE_integer(flag_name='test_batch_size', default_value=5000, docstring='test_batch_size')



