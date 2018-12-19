import tensorflow as tf
import numpy as np
import os


# 读取数据，制作图片list和标签list
img_width = 208 # resize the image, if the input image is too large, training will be very slow.
img_height = 208
train_dir = 'data/train'
N_CLASSES = 2
BATCH_SIZE = 16
CAPACITY = 2000 # 队列中元素个数
MAX_STEP = 10000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001

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
    image_batch = tf.cast(image, tf.float32)
    # 如果图片太大需要裁剪crop，太小需要扩充pad图片
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 在把数据送进神经网络之前，需要将数据图片标准化，减去均值除以方差
    image = tf.image.per_image_standardization(image)
    #image = image / 255

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size= batch_size,
                                              num_threads= 64,
                                              capacity= capacity)

    label_batch = tf.reshape(label_batch, [batch_size]) # 把label_batch变成batch_size这么多行的一个tensor（张量）

    return image_batch, label_batch

def model(images, batch_size, n_classes):
    '''Build the model
    n_classes = 2, 因为是有猫和狗
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''

    # conv1, shape = [kernel size, kernel size, channels, kernel numbers] [前两个是卷积盒的大小3*3，RGB，卷积盒的数量（输出的个数）]
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')  # images和weights做卷积
        pre_activation = tf.nn.bias_add(conv, biases)  # 把卷积结果和biases加在一起
        conv1 = tf.nn.relu(pre_activation, name=scope.name)  # relu做激活函数，得多卷积层1的输出

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],  # 输入value是conv1
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')
    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')

    # local3 全连接层
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value  # 这两步是把pool2的结果拉平成向量
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],  # 神经元128个
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32))  # 初始值0.005
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))  # biases 初始值依然是0.1
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        local3 = tf.nn.dropout(local3, 0.7)

        # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],  # 第四层，即第二层全连接层还是128
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
        local4 = tf.nn.dropout(local4,0.7)

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],   # 输入128，最后输出两个类
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')  # 最后点乘后加上biases，返回
        # 此处没必要加激活函数，因为在后面的losses里面包含了
    return softmax_linear


def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]   inference的返回值softmax_linear
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')   # 用sparse_就不用one hot，否则需要
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()
        learning_rate 学习率
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # adam的优化器
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)    # 优化loss
    return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)   # in_top_k 是拿logits和label作比较，只取1的，看预测的最大值的那个label和真实的label是不是一样
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)    # 取均值
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

################################## 执行：开始训练，并保存模型和数据，测试时把这段注释掉即可 ################################
# logs_train_dir = 'cats_vs_dogs/mylogs'
#
# train_image, train_label = get_files(train_dir)
# image_batch, label_batch = get_batch(train_image, train_label, img_width, img_height, BATCH_SIZE, CAPACITY)
# print(image_batch, label_batch)
#
# softmax_linear = model(image_batch,BATCH_SIZE,N_CLASSES)
# loss = losses(softmax_linear,label_batch)
# train_op = trainning(loss,learning_rate)
# accuracy = evaluation(softmax_linear,label_batch)
#
# summary_op = tf.summary.merge_all()  # 把所有的summary合并到一块
# sess = tf.Session()
# train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)  # 把summary保存到路径中
# saver = tf.train.Saver()
#
# sess.run(tf.global_variables_initializer())
# coord = tf.train.Coordinator()  # 开启批次监控
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
# try:
#     for step in np.arange(MAX_STEP):  # 定义一个循环终止条件
#         if coord.should_stop():
#             break
#         _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy])  # 如果能正常运行
#
#         if step % 50 == 0:  # 每50步print一下，保存一下summary
#             print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
#             summary_str = sess.run(summary_op)
#             train_writer.add_summary(summary_str, step)
#
#         if step % 2000 == 0 or (step + 1) == MAX_STEP:  # 每两千步保存一下
#             checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
#             saver.save(sess, checkpoint_path, global_step=step)
#
# except tf.errors.OutOfRangeError:
#     print('Done training -- epoch limit reached')
# finally:
#     coord.request_stop()  # 结束
#
# coord.join(threads)
# sess.close()
########################################################################################################################

############# 测试 ####################

from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   plt.imshow(image)
   image = image.resize([208, 208])
   image = np.array(image)
   return image

def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   test_dir = 'data/test'
   train, train_label = get_files(test_dir)
   image_array = get_one_image(train)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 208, 208, 3])
       logit = model(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[208, 208, 3])

       # you need to change the directories to yours.
       logs_train_dir = 'cats_vs_dogs/mylogs'

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a cat with possibility %.6f' %prediction[:, 0])
           else:
               print('This is a dog with possibility %.6f' %prediction[:, 1])
   plt.show()

evaluate_one_image()