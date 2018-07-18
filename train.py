import model as alexnet
import cifar10_input as loader
import tensorflow as tf
import numpy as np
import resource, sys, timeit
from datetime import datetime

import argparse

def main():
    _IMG_SIZE = 32
    # _IMG_SIZE = 28
    _IMG_CHANNEL = 3
    _IMG_CLASS = 10

    parser = argparse.ArgumentParser(description='alexnet')
    parser.add_argument('--data_dir', type=str, default='./cifar-10-batches-py/')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=100) # must be integer times of total images_num
    parser.add_argument('--keepPro', type=float, default=0.5)
    parser.add_argument('--summary_dir', type=str, default='./summary/alexnetlog/')
    parser.add_argument('--max_epoch', type=int, default=40)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    args = parser.parse_args()
    print(args)
    with tf.device('/gpu:1'):
        # -----------------------------------------------------------------------------
        #       BUILD GRAPH
        # -----------------------------------------------------------------------------
        inputs, labels, dropout_keep_prob, learning_rate, is_training = alexnet.input_placeholder(_IMG_SIZE, _IMG_CHANNEL, _IMG_CLASS)
        logits = alexnet.interface(inputs, args.keepPro, _IMG_CLASS, is_training)

        accuracy = alexnet.accuracy(logits, labels)
        loss = alexnet.loss(logits, labels)
        train = alexnet.train(loss, learning_rate, 'RMSProp')
        
        # ready for summary or save
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        print("[BUILD GRAPH] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)

        # -----------------------------------------------------------------------------
        #       LOAD DATA
        # -----------------------------------------------------------------------------
        train_images, train_labels, test_images, test_labels = loader.load_batch_data(args.data_dir, args.batch_size)
        print("[LOAD DATA] memory_usage=%f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), file=sys.stderr)

        # -----------------------------------------------------------------------------
        #       START THE SESSION
        # -----------------------------------------------------------------------------
        cur_lr = args.learning_rate # current learning rate
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(logdir=args.summary_dir + 'norm' + '/train/', 
                                                 graph=sess.graph)
            test_writer = tf.summary.FileWriter(logdir=args.summary_dir + 'norm' + '/test/')

            for epoch in range(args.max_epoch):
                start = timeit.default_timer()
                train_accs = []
                train_losses = []
                # train
                for index, images_batch in enumerate(train_images):
                    _, summary, train_loss, train_acc = sess.run(fetches = [train, merged, loss, accuracy], 
                                                                 feed_dict = { inputs: images_batch,
                                                                               labels: train_labels[index],
                                                                               dropout_keep_prob: args.keepPro,
                                                                               learning_rate: cur_lr,
                                                                               is_training: True })
                    train_accs.append(train_acc)
                    train_losses.append(train_loss)
                    print('[batch] {} done'.format(index), end='\r')
                train_avg_acc  = float(np.mean(np.asarray(train_accs)))
                train_avg_loss = float(np.mean(np.asarray(train_losses)))
                train_summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=train_avg_acc), 
                                                  tf.Summary.Value(tag="avg_loss", simple_value=train_avg_loss),
                                                  tf.Summary.Value(tag="learning_rate", simple_value=cur_lr)])
                train_writer.add_summary(summary, epoch)
                train_writer.add_summary(train_summary, epoch)
                print('=' * 20 + 'EPOCH {} [TRAIN]'.format(epoch) + '=' * 20)
                print('cost time: {:.3f}s'.format(timeit.default_timer()-start))
                print('acc: {0}, avg_loss: {1}'.format(train_avg_acc, train_avg_loss))
                # evaluate
                if (epoch + 1) % args.eval_freq == 0:
                    test_accs = []
                    test_losses = []
                    for index, test_images_batch in enumerate(test_images):
                        test_loss, test_acc = sess.run(fetches = [loss, accuracy], 
                                                       feed_dict = { inputs: test_images_batch,
                                                                     labels: test_labels[index],
                                                                     dropout_keep_prob: args.keepPro,
                                                                     learning_rate: cur_lr,
                                                                     is_training: False})
                        test_accs.append(test_acc)
                        test_losses.append(test_loss)
                    test_avg_acc  = float(np.mean(np.asarray(test_accs)))
                    test_avg_loss = float(np.mean(np.asarray(test_losses)))
                    test_summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=test_avg_acc), 
                                                     tf.Summary.Value(tag="avg_loss", simple_value=test_avg_loss)])
                    test_writer.add_summary(test_summary, epoch)
                    print('=' * 20 + 'EPOCH {} [EVAL]'.format(epoch) + '=' * 20)
                    print('acc: {0}, avg_loss: {1}'.format(test_avg_acc, test_avg_loss))
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
    # if (epoch + 1) == 5:
    #     lr = lr * 0.1
    # if (epoch + 1) == 10:
    #     lr = lr * 0.1
    # # if (epoch + 1) == 30:
    # #     lr = lr * 0.1
    # if epoch > 10 and (epoch + 1) % 10 == 0:
    #     lr = lr * 0.8
    return lr

if __name__ == '__main__':
    main()