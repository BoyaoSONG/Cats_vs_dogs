import tensorflow as tf
import numpy as np
import os
import math

file_dir = train_dir = 'small_data/train'
image_W = 208
image_H = 208
batch_size = 16 # batch_size<64比较好，内存内不报错就好，越大图上越不震荡
capacity = 100
n_classes = 2
learning_rate = 0.0001
logs_train_dir = 'small_data/logs/train'
logs_var_dir = 'small_data/logs/validation'
ratio = 0.2     # 取20%做验证

def get_files(file_dir,ratio):      # ratio 是说在原本的数据中，拿出百分之多少做validation（validation的内容是训练是机器完全没有见过的）
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
    temp = temp.transpose() # 转置成列向量了
    np.random.shuffle(temp) # 打乱我们图片以及其对应的标签

    all_image_list = list(temp[:,0])    # 再重新读回来图片,读取第0列，返回成行向量
    all_label_list = list(temp[:,1])    # 再重新读回来标签，,读取第1列，返回成行向量

    n_sample = len(all_label_list)  # 总的图片的个数
    n_val = math.ceil(n_sample*ratio)   # number of validation（ceil取整数）
    n_train = n_sample - n_val  # 训练集数据个数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int((float(i))) for i in tra_labels]
    val_images = all_image_list[n_train:]
    val_labels = all_label_list[n_train:]
    val_labels = [int(float(i)) for i in val_labels]

    return  tra_images,tra_labels,val_images,val_labels



def get_batch(image_list, label_list, image_W, image_H, batch_size, capacity):
    image = tf.cast(image_list, tf.string)  # 用cast把list的格式转换为TensorFlow可以识别的格式，转化数据类型
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # image需要用一个TensorFlow的reader来读取出来
    image = tf.image.decode_jpeg(image_contents, channels=3)  # 用jpeg解码器把image解码
# fix me
    image_batch = tf.cast(image, tf.float32)
    # 如果图片太大需要裁剪crop，太小需要扩充pad图片
    image = tf.image.resize_image_with_crop_or_pad(image_batch, image_W, image_H)
    # 在把数据送进神经网络之前，需要将数据图片标准化，减去均值除以方差
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])  # 把label_batch变成batch_size这么多行的一个tensor（张量）

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
                                  shape=[5, 5, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')  # images和weights做卷积
        pre_activation = tf.nn.bias_add(conv, biases)  # 把卷积结果和biases加在一起   208*208*16
        conv1 = tf.nn.relu(pre_activation, name=scope.name)  # relu做激活函数，得多卷积层1的输出

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  # 输入value是conv1
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')
    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 16, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  # 输入value是conv1
                               padding='SAME', name='pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 32, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm2, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv3')

    # pool3 and norm3
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  # 输入value是conv1
                               padding='SAME', name='pooling2')
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm3')


    # conv4
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 64, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm3, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name='conv4')

    # pool4 and norm4
    with tf.variable_scope('pooling4_lrn') as scope:
        norm4 = tf.nn.lrn(conv4, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        pool4 = tf.nn.max_pool(norm4, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling4')

    # local3 全连接层
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool4, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value  # 这两步是把pool3的结果拉平成向量
        weights = tf.get_variable('weights',
                                  shape=[dim, 256],  # 神经元128个
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32))  # 初始值0.005
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))  # biases 初始值依然是0.1
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)


        # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[256, 256],  # 第四层，即第二层全连接层还是128
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
        local4 = tf.nn.dropout(local4, 0.7)


    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[256, n_classes],   # 输入256，最后输出两个类
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')  # 最后点乘后加上biases，返回
        # 此处没必要加激活函数，因为在后面的losses里面包含了
    return softmax_linear

def losses(softmax_linear, labels):
    '''Compute loss from logits and labels
    Args:
        softmax_linear: logits tensor, float, [batch_size, n_classes]   inference的返回值softmax_linear
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=softmax_linear, labels=labels, name='xentropy_per_example')   # 用sparse_就不用one hot，否则需要
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


# tra_images,tra_labels,val_images,val_labels = get_files(file_dir,ratio)
# tra_image_batch, tra_label_batch = get_batch(tra_images,tra_labels,image_W,image_H,batch_size,capacity)
# val_image_batch, val_label_batch = get_batch(val_images,val_labels,image_W,image_H,batch_size,capacity)
#
#
# softmax_linear = model(tra_image_batch,batch_size,n_classes)
# loss = losses(softmax_linear,tra_label_batch)
# train_op = trainning(loss,learning_rate)
# accuracy = evaluation(softmax_linear,tra_label_batch)
#
# x = tf.placeholder(tf.float32, shape=[batch_size,image_W,image_H,3])
# y_ = tf.placeholder(tf.int16, shape=[batch_size])
#
# with tf.Session() as sess:
#     saver = tf.train.Saver()
#     sess.run(tf.global_variables_initializer())
#     coord = tf.train.Coordinator()  # 开启批次监控
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     summary_op = tf.summary.merge_all()
#     train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
#     val_writer = tf.summary.FileWriter(logs_var_dir, sess.graph)
#
# # 开始训练
#     try:
#         for step in np.arange(500):  # 定义一个循环终止条件，1000次训练
#             if coord.should_stop():
#                 break
#
#             tra_images,tra_labels = sess.run([tra_image_batch,tra_label_batch])
#             _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],feed_dict={x:tra_images,y_:tra_labels})  # 如果能正常运行
#
#             if step % 50 == 0:  # 每50步print一下，保存一下summary
#                 print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
#                 summary_str = sess.run(summary_op)
#                 train_writer.add_summary(summary_str, step)
#
#             if step % 50 == 0 or (step+1) == 500 :
#                 val_images,val_labels = sess.run([val_image_batch,val_label_batch])
#                 val_loss, val_acc = sess.run([loss,accuracy],feed_dict={x:val_images,y_:val_labels})
#                 print('Step %d, validation loss = %.2f, validation accuracy = %.2f%%' % (step, val_loss, val_acc * 100.0))
#                 summary_str = sess.run(summary_op)
#                 val_writer.add_summary(summary_str, step)
#
#             if step % 100 == 0 or (step + 1) == 500:  # 每200步保存一下
#                 checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
#                 saver.save(sess, checkpoint_path, global_step=step)
#
#     except tf.errors.OutOfRangeError:
#         print('Done training -- epoch limit reached')
#     finally:
#         coord.request_stop()  # 结束
#
#     coord.join(threads)
#     sess.close()



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
    test_dir = 'small_data/test'
    test = []
    for file in os.listdir(test_dir):

        test.append(test_dir+'/'+file)

    image_array = get_one_image(test)

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
       logs_train_dir = 'small_data/logs/train'

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