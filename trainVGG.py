from vgg16 import vgg16
import cifar10_input as loader
import tensorflow as tf
import numpy as np
import resource, sys
from datetime import datetime

import argparse

def main():
    # _IMG_SIZE = 32
    _IMG_SIZE = 28
    _IMG_CHANNEL = 3
    _IMG_CLASS = 10

    parser = argparse.ArgumentParser(description='vgg16')
    parser.add_argument('--data_dir', type=str, default='./cifar-10-batches-py/')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=100) # must be integer times of total images_num
    parser.add_argument('--summary_dir', type=str, default='./summary/vgglog/')
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=300)
    args = parser.parse_args()
    print(args)
    with tf.device('/gpu:1'):
        # -----------------------------------------------------------------------------
        #       BUILD GRAPH
        # -----------------------------------------------------------------------------
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, _IMG_SIZE, _IMG_SIZE, _IMG_CHANNEL], name='inputs')
        labels = tf.placeholder(dtype=tf.int64, shape=None, name='labels')
        learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
        keepPro = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')

        model = vgg16(imgs=inputs, weights=args.data_dir + 'vgg16_weights.npz', is_training=is_training, keepPro=keepPro)
        logits = model.probs
        # loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                                                 labels=labels, 
                                                                                 name='cross_entropy'))
        tf.summary.scalar('loss', loss)
        # optimize
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            with tf.name_scope("train"):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, 
                                                    decay=0.9, 
                                                    momentum=0.0, 
                                                    epsilon=1e-10, 
                                                    use_locking=False, 
                                                    name='RMSProp').minimize(loss)
        # evaluate acc
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        
        # ready for summary or save
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        print("[BUILD GRAPH] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)

        # -----------------------------------------------------------------------------
        #       LOAD DATA
        # -----------------------------------------------------------------------------
        train_images, train_labels, test_images, test_labels = loader.load_batch_data_aug(args.data_dir, args.batch_size)
        print("[LOAD DATA] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)

        # -----------------------------------------------------------------------------
        #       START THE SESSION
        # -----------------------------------------------------------------------------
        cur_lr = args.learning_rate # current learning rate
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(logdir=args.summary_dir + datetime.now().strftime('%Y%m%d-%H%M%S') + '/train/', 
                                                 graph=sess.graph)
            test_writer = tf.summary.FileWriter(logdir=args.summary_dir + datetime.now().strftime('%Y%m%d-%H%M%S') + '/test/')

            for epoch in range(args.max_epoch):
                train_accs = []
                train_losses = []
                # train
                for index, images_batch in enumerate(train_images):
                    _, summary, train_loss, train_acc = sess.run(fetches = [optimizer, merged, loss, accuracy], 
                                                                 feed_dict = { inputs: images_batch,
                                                                               labels: train_labels[index],
                                                                               learning_rate: cur_lr,
                                                                               is_training: True,
                                                                               keepPro: 0.6})
                    train_accs.append(train_acc)
                    train_losses.append(train_loss)
                    # print('[batch] {} done'.format(index), end='\r')
                train_avg_acc  = float(np.mean(np.asarray(train_accs)))
                train_avg_loss = float(np.mean(np.asarray(train_losses)))
                train_summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=train_avg_acc), 
                                                  tf.Summary.Value(tag="loss", simple_value=train_avg_loss),
                                                  tf.Summary.Value(tag="learning_rate", simple_value=cur_lr)])
                train_writer.add_summary(summary, epoch)
                train_writer.add_summary(train_summary, epoch)
                print('=' * 20 + 'EPOCH {} [TRAIN]'.format(epoch) + '=' * 20)
                print('acc: {0}, loss: {1}'.format(train_avg_acc, train_avg_loss))
                # evaluate
                if (epoch + 1) % args.eval_freq == 0:
                    test_accs = []
                    test_losses = []
                    for index, test_images_batch in enumerate(test_images):
                        test_loss, test_acc = sess.run(fetches = [loss, accuracy], 
                                                       feed_dict = { inputs: test_images_batch,
                                                                     labels: test_labels[index],
                                                                     learning_rate: cur_lr,
                                                                     is_training: False,
                                                                     keepPro: 0.6})
                        test_accs.append(test_acc)
                        test_losses.append(test_loss)
                    test_avg_acc  = float(np.mean(np.asarray(test_accs)))
                    test_avg_loss = float(np.mean(np.asarray(test_losses)))
                    test_summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=test_avg_acc), 
                                                     tf.Summary.Value(tag="loss", simple_value=test_avg_loss)])
                    test_writer.add_summary(test_summary, epoch)
                    print('=' * 20 + 'EPOCH {} [EVAL]'.format(epoch) + '=' * 20)
                    print('acc: {0}, loss: {1}'.format(test_avg_acc, test_avg_loss))
                # lr decay
                cur_lr = lr(cur_lr, epoch)
                # save
                if (epoch + 1) % args.save_freq == 0:
                    checkpoint_file = args.summary_dir + 'model.ckpt'
                    saver.save(sess, checkpoint_file, global_step=epoch)
                    print('Saved checkpoint')
            
            train_writer.close()
            test_writer.close()

def lr(lr, epoch):
    lr = lr * 0.9
    return lr

if __name__ == '__main__':
    main()