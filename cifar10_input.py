import pickle
import numpy as np

def load_CIFAR_batch(filename):
    """load single batch of cifar
    Args:
        filename: the path of file
    Returns:
        X: images with shape (10000, 32, 32, 3)
        Y: labels 0-9 with shape(10000,)
    """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        # X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        X = X.astype(float)
        # X = X.reshape(10000, 3 * 32 * 32)
        Y = np.array(Y)
        return X, Y

def load_batch_data(data_dir, batch_size):
    """load all data and devide it to mini-batch
    Args:
        data_dir: the parent directory path
        batch_size: the size of mini-batch while training
    Returns: 
        train_images, train_labels: <list> devided to mini-batch
        test_images, test_labels: <nparray>

    """
    train_images = []
    train_labels = []

    _CIFAR_BATCH = 5
    for i in range(1, _CIFAR_BATCH + 1):
        filename = data_dir + 'data_batch_{}'.format(i)
        X, Y = load_CIFAR_batch(filename)
        train_images.append(X)
        train_labels.append(Y)

    train_images = np.concatenate(train_images) # with shape(50000, -1)
    train_labels = np.concatenate(train_labels) # with shape(50000,)

    test_images, test_labels = load_CIFAR_batch(data_dir + 'test_batch') # with shape(10000, -1) and (10000,)

    # normalize
    mean = np.mean(train_images, axis=0)
    std = np.std(train_images, axis=0)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    # reshape
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0,2,3,1) # with shape(50000, 32, 32 3)
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0,2,3,1) # with shape(10000, 32, 32, 3)

    # devide to mini-batch
    train_images = np.split(train_images, _CIFAR_BATCH * 10000 // batch_size, axis=0)
    train_labels = np.split(train_labels, _CIFAR_BATCH * 10000 // batch_size, axis=0)
    test_images = np.split(test_images, 10000 // batch_size, axis=0)
    test_labels = np.split(test_labels, 10000 // batch_size, axis=0)

    return train_images, train_labels, test_images, test_labels

def load_batch_data_aug(data_dir, batch_size):
    """load all data and make data augmentation and devide it to mini-batch
    Args:
        data_dir: the parent directory path
        batch_size: the size of mini-batch while training
    Returns: 
        train_images, train_labels: <list> devided to mini-batch
        test_images, test_labels: <nparray>

    """
    train_images = []
    train_labels = []

    _CIFAR_BATCH = 5
    for i in range(1, _CIFAR_BATCH + 1):
        filename = data_dir + 'data_batch_{}'.format(i)
        X, Y = load_CIFAR_batch(filename)
        train_images.append(X)
        train_labels.append(Y)

    train_images = np.concatenate(train_images) # with shape(50000, -1)
    train_labels = np.concatenate(train_labels) # with shape(50000,)

    test_images, test_labels = load_CIFAR_batch(data_dir + 'test_batch') # with shape(10000, -1) and (10000,)

    # normalize
    mean = np.mean(train_images, axis=0)
    std = np.std(train_images, axis=0)
    std = 1
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    # reshape
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0,2,3,1) # with shape(50000, 32, 32 3)
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0,2,3,1) # with shape(10000, 32, 32, 3)

    #augmentation
    x_crop_top_left = train_images[:,:28,:28,:]
    x_crop_bottom_right = train_images[:,4:,4:,:]
    x_flip_top_left = x_crop_top_left[:,:,::-1,:]
    x_flip_bottom_right = x_crop_bottom_right[:,:,::-1,:]
    train_images = np.concatenate((x_crop_top_left, x_crop_bottom_right, 
                                   x_flip_top_left, x_flip_bottom_right), axis=0) # with shape(200000, 28, 28 3)
    train_labels = np.concatenate((train_labels, train_labels, train_labels, train_labels), axis=0)

    test_images = test_images[:,2:30,2:30,:]

    # devide to mini-batch
    train_images = np.split(train_images, _CIFAR_BATCH * 10000 * 4 // batch_size, axis=0)
    train_labels = np.split(train_labels, _CIFAR_BATCH * 10000 * 4// batch_size, axis=0)
    test_images = np.split(test_images, 10000 // batch_size, axis=0)
    test_labels = np.split(test_labels, 10000 // batch_size, axis=0)

    return train_images, train_labels, test_images, test_labels

def main():
    """test for loading data"""
    data_dir = './cifar-10-batches-py/'
    # train_images, train_labels, test_images, test_labels = load_batch_data(data_dir, 5000)
    train_images, train_labels, test_images, test_labels = load_batch_data_aug(data_dir, 10000)
    for index, images_batch in enumerate(train_images):
        print('images_batch {0}: {1}'.format(index, images_batch.shape))
        print('labels_batch {0}: {1}'.format(index, train_labels[index].shape))
    for index, test_images_batch in enumerate(test_images):
        print('test images_batch {0}: {1}'.format(index, test_images_batch.shape))
        print('test labels_batch {0}: {1}'.format(index, test_labels[index].shape))

if __name__ == '__main__':
    main()