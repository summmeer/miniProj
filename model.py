import tensorflow as tf
import numpy as np
import math

# -----------------------------------------------------------------------------
#        OPS FOR NET
# -----------------------------------------------------------------------------
def maxpool(x, ksize, strides, padding = "SAME"):
    """max-pooling layer"""
    return tf.nn.max_pool(x, 
                          ksize = [1, ksize, ksize, 1], 
                          strides = [1, strides, strides, 1], 
                          padding = padding, 
                          name='maxpooling')

def dropout(x, keepPro):
    """drop out layer"""
    return tf.nn.dropout(x, keepPro, name='dropout')

def lrn(x, depth_r=2, alpha=0.0001, beta=0.75, bias=1.0):
    """local response normalization"""
    return tf.nn.local_response_normalization(x, 
                                              depth_radius = depth_r, 
                                              alpha = alpha, 
                                              beta = beta, 
                                              bias = bias, 
                                              name='lrn')

def fc(x, output_size, name, activation_func=tf.nn.relu):
    """fully connected layer"""
    with tf.variable_scope(name):
        input_size = x.get_shape().as_list()[-1]
        w = tf.Variable(tf.random_normal([input_size, output_size], 
                        dtype=tf.float32, 
                        stddev=0.01), 
                        name='weights')
        b = tf.Variable(tf.constant(value=0.0, 
                        dtype = tf.float32, 
                        shape=[output_size]), 
                        name='bias')
        # another way to init
        # bound = math.sqrt(6 / input_size + output_size)
        # w = tf.get_variable(name='weights',
        #                     shape=[input_size, output_size],
        #                     initializer=tf.random_uniform_initializer(minval=-bound,
        #                                                             maxval=bound))
        # bound = 1 / input_size
        # b = tf.get_variable(name='bias',
        #                     shape=[output_size],
        #                     initializer=tf.random_uniform_initializer(minval=-bound,
        #                                                             maxval=bound))
        out = tf.nn.xw_plus_b(x, w, b)
        if activation_func:
            return activation_func(out)
            tf.summary.histogram('fc', out)
        else:
            # return tf.nn.softmax(out)
            return out

def conv(x, ksize, strides, output_size, name, activation_func=tf.nn.relu, padding = "SAME", bias=0.0):
    """conv layer"""
    with tf.variable_scope(name):
        input_size = x.get_shape().as_list()[-1]
        w = tf.Variable(tf.random_normal([ksize, ksize, input_size, output_size], 
                        dtype=tf.float32, 
                        stddev=0.01), 
                        name='weights')
        b = tf.Variable(tf.constant(value=bias, 
                        dtype=tf.float32, 
                        shape=[output_size]), 
                        name='bias')
        # another way to init
        # bound = math.sqrt(6 / input_size + output_size)
        # w = tf.get_variable(name='weights',
        #                     shape=[ksize, ksize, input_size, output_size],
        #                     initializer=tf.random_uniform_initializer(minval=-bound,
        #                                                             maxval=bound))
        # bound = 1 / input_size
        # b = tf.get_variable(name='bias',
        #                     shape=[output_size],
        #                     initializer=tf.random_uniform_initializer(minval=-bound,
        #                                                             maxval=bound))
        conv = tf.nn.conv2d(x, w, [1, strides, strides, 1], padding=padding)
        conv = tf.nn.bias_add(conv, b)
        tf.summary.histogram('conv', conv)
        if activation_func:
            conv = activation_func(conv)
            tf.summary.histogram('conv_relu',conv)
        return conv

# -----------------------------------------------------------------------------
#        ALEXNET MODEL
# -----------------------------------------------------------------------------
# IMG_SIZE = 32
# IMG_CHANNEL = 3
# IMG_CLASS = 10

def interface(x, keepPro, class_num, is_training):
    """build alexnet
    Args: 
        x: NWHC input feature map
        keepPro: dropout ratio
        class_num: the number of class
        is_training: bool flag of training(required by batch-normalization)
    Returns:
        score of fc output
    """
    # layer 1
    with tf.name_scope('conv1layer'):
        conv1 = conv(x=x, ksize=5, strides=1, output_size=48, name='conv1') # output[32,32,48]
        conv1 = lrn(conv1)
        print('conv1:{}'.format(conv1.get_shape().as_list()))
        conv1 = maxpool(conv1, ksize=3, strides=2, padding='VALID') # output[15,15,48]
        print('maxpool1:{}'.format(conv1.get_shape().as_list()))
        conv1 = batch_norm(conv1, 'bn1', is_training)
    
    # layer 2
    with tf.name_scope('conv2layer'):
        conv2 = conv(x=conv1, ksize=5, strides=1, output_size=128, bias=1.0, name='conv2') # output[15,15,128]
        conv2 = lrn(conv2)
        print('conv2:{}'.format(conv2.get_shape().as_list()))
        conv2 = maxpool(conv2, ksize=3, strides=2, padding='VALID') # output[7,7,128]
        print('maxpool2:{}'.format(conv2.get_shape().as_list()))
        conv2 = batch_norm(conv2, 'bn2', is_training)
    
    # layer 3
    with tf.name_scope('conv3layer'):
        conv3 = conv(x=conv2, ksize=3, strides=1, output_size=192, name='conv3') # output[7,7,192]
        print('conv3:{}'.format(conv3.get_shape().as_list()))
        conv3 = batch_norm(conv3, 'bn3', is_training)
    
    # layer 4
    with tf.name_scope('conv4layer'):
        conv4 = conv(x=conv3, ksize=3, strides=1, output_size=192, bias=1.0, name='conv4') # output[7,7,192]
        print('conv4:{}'.format(conv4.get_shape().as_list()))
        conv4 = batch_norm(conv4, 'bn4', is_training)
    
    # layer 5
    with tf.name_scope('conv5layer'):
        conv5 = conv(x=conv4, ksize=3, strides=1, output_size=128, bias=1.0, name='conv5') # output[7,7,128]
        print('conv5:{}'.format(conv5.get_shape().as_list()))
        conv5 = maxpool(conv5, ksize=3, strides=2, padding='VALID') #output[3,3,128]
        print('maxpool5:{}'.format(conv5.get_shape().as_list()))
        conv5 = batch_norm(conv5, 'bn5', is_training)
    
    # flatten
    conv5size = conv5.get_shape().as_list()
    conv5 = tf.reshape(conv5, [-1, conv5size[1] * conv5size[2] * conv5size[3]])
    print('flatten:{}'.format(conv5.get_shape().as_list()))

    # layer 6
    with tf.name_scope('fc1layer'):
        fc1 = fc(x=conv5, output_size=512, name='fc1')
        print('fc1:{}'.format(fc1.get_shape().as_list()))
        fc1 = dropout(fc1, keepPro)
        fc1 = batch_norm(fc1, 'bn6', is_training)

    # layer 7
    with tf.name_scope('fc2layer'):
        fc2 = fc(x=fc1, output_size=256, name='fc2')
        print('fc2:{}'.format(fc2.get_shape().as_list()))
        fc2 = dropout(fc2, keepPro)
        fc2 = batch_norm(fc2, 'bn7', is_training)

    # layer 8 - output
    with tf.name_scope('fc3layer'):
        return fc(x=fc2, output_size=class_num, activation_func=None, name='fc3')

def batch_norm(x, name, is_training):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x, axis=-1, training=is_training)

def input_placeholder(img_size, img_channel, class_num):
    with tf.name_scope('inputlayer'):
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, img_channel], name='inputs')
        labels = tf.placeholder(dtype=tf.int64, shape=None, name='labels')
    
    dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    return inputs, labels, dropout_keep_prob, learning_rate, is_training

def accuracy(logits, labels):
    """compute accuracy"""
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy

def loss(logits, labels):
    """compute loss"""
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                                             labels=labels, 
                                                                             name='cross_entropy'))
        tf.summary.scalar('loss', loss)
    return loss

def train(loss, learning_rate, optimizer='RMSProp'):
    """
    train model to minimize loss
    Args:
        loss: softmax_cross_entropy loss
        learning_rate: learning rate (decay by hand)
        optimizer: 'RMSProp' or 'AdamProp'
    """
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # required by batch normalization
    with tf.control_dependencies(update_ops):
        if optimizer == 'RMSProp':
            return tf.train.RMSPropOptimizer(learning_rate=learning_rate, 
                                                decay=0.9, 
                                                momentum=0.0, 
                                                epsilon=1e-10, 
                                                use_locking=False, 
                                                name='RMSProp').minimize(loss)
        if optimizer == 'AdamProp':
            return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08, 
                                        name='AdamProp').minimize(loss)