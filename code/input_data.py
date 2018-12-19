"""
本代码用来读取数据
生成批次

"""
import tensorflow as tf
import numpy as np
import os

img_width = 208
img_height = 208

train_dir = 'data/train'

def get_files(file_dir):
    """
    :param
        file_dir: file directory
    :return:
        list of images and labels
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0]=='cat':
            cats.append(file_dir + '/' + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + '/' + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats),len(dogs)))

    image_list = np.hstack((cats,dogs))     # 将所有猫和狗的照片堆叠
    label_list = np.hstack((label_cats, label_dogs))    # 将其对应的label也堆叠在一起

    temp = np.array([image_list, label_list])
    temp = temp.transpose() # 转置
    np.random.shuffle(temp) # 打乱我们图片以及其对应的标签

    image_list = list(temp[:,0])    # 再重新读回来图片
    label_list = list(temp[:,1])    # 再重新读回来标签
    label_list = [int(i) for i in label_list]

    return  image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size  每一批一批多少个图片
        capacity: the maximum element in queue  在队列中最多能够容大的元素个数
    :return:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    image = tf.cast(image, tf.string)   # 用cast把list的格式转换为TensorFlow可以识别的格式，转化数据类型
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    # slice_input_producer（list，epoch）生成一个队列
    # 把image和label组成一个list给进来，epoch如果不指定，将会在队列中一直有元素，会循环无数次，出去10个元素，进来10个元素

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])   # image需要用一个TensorFlow的reader来读取出来
    image = tf.image.decode_jpeg(image_contents, channels=3)    # 用jpeg解码器把image解码
    # The image from your input pipeline is of type 'uint8', you need to type cast it to 'float32', You can do this after the image jpeg decoder
    image = tf.cast(image, tf.float32)
    # 如果图片太大需要裁剪crop，太小需要扩充pad图片
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 在把数据送进神经网络之前，需要将数据图片标准化，减去均值除以方差
    #image = tf.image.per_image_standardization(image)
    #image = image / 255

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size= batch_size,
                                              num_threads= 64,
                                              capacity= capacity)
    label_batch = tf.reshape(label_batch, [batch_size]) # 把label_batch变成batch_size这么多行的一个tensor（张量）

    return image_batch, label_batch

## Test

# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208
#
# train_dir = 'data/train'
#
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    try:
#        while not coord.should_stop() and i<1:
#
#            img, label = sess.run([image_batch, label_batch])
#
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)
