# coding: utf-8
import tensorflow as tf
import os
import pickle
import tensorflow.contrib.slim as slim
import random
import numpy
import cv2

tf.app.flags.DEFINE_string("mode", "test", "train or test or inside or evaluation")
tf.app.flags.DEFINE_string("checkpoint", "./checkpoint/", "dir of checkpoint")
tf.app.flags.DEFINE_string("train_dir", "./dataset/train", "dir of training data")
tf.app.flags.DEFINE_string("test_dir", "./dataset_auto/test", "dir of test data")  # ./dataset_auto/test or ./dataset/test
tf.app.flags.DEFINE_string("logger_dir", "./logger", "dir of logger")
tf.app.flags.DEFINE_integer("batch_size", 128, "size of batch")
tf.app.flags.DEFINE_integer("img_size", 64, "size of resized images")
tf.app.flags.DEFINE_string("char_dict", "char_dict", "path to character dict")
tf.app.flags.DEFINE_bool("restore", False, "restore from previous checkpoint")
tf.app.flags.DEFINE_integer("max_step", 100001, "maximum steps")
tf.app.flags.DEFINE_string("temp", "./temp", "path to test picture")
FLAGS = tf.app.flags.FLAGS


# 从路径读取数据
def get_image_path_and_labels(dir):
    img_path = []
    # 遍历图片保存路径
    for root, dir, files in os.walk(dir):
        img_path += [os.path.join(root, f) for f in files]
    # 打乱图像列表
    random.shuffle(img_path)
    # 生成图片label
    labels = [int(name.split(os.sep)[1]) for name in img_path]
    return img_path, labels


# 生成batch
def batch(dir, batch_size, prepocess=False):
    img_path, labels = get_image_path_and_labels(dir)
    # 将数据转变为tensor数据
    img_tensor = tf.convert_to_tensor(img_path, dtype=tf.string)
    lb_tensor = tf.convert_to_tensor(labels, dtype=tf.int64)
    # 以切片的方式分批读取数据
    input_pipe = tf.train.slice_input_producer([img_tensor, lb_tensor])
    # 读取普片数据并转为灰度
    img = tf.read_file(input_pipe[0])
    imgs = tf.image.convert_image_dtype(tf.image.decode_png(img, channels=1), tf.float32)
    # 随机修改图像，为了避免过度拟合
    if prepocess:
        imgs = tf.image.random_contrast(imgs, 0.9, 1.1)
    # 图片变为统一大小
    imgs = tf.image.resize_images(imgs, tf.constant([FLAGS.img_size, FLAGS.img_size], dtype=tf.int32))
    # 读取label
    lbs = input_pipe[1]
    # batch=batch_size
    img_batch, lb_batch = tf.train.shuffle_batch([imgs, lbs], batch_size=batch_size, capacity=50000,
                                                 min_after_dequeue=10000)
    return img_batch, lb_batch


def cnn():
    # 数据预加载，传入数据用feed_dict
    # 保留比例
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    # 图片数据
    img = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="img_batch")
    # 标签
    labels = tf.placeholder(tf.int64, shape=[None], name="label_batch")

    # Structure references to : http://yuhao.im/files/Zhang_CNNChar.pdf,
    # 卷积神经网络的结构

    # 第一层卷积，卷积核大小3*3， 输出个数64， 默认步长为1， 填充方式为SAME
    conv1 = slim.conv2d(img, 64, [3, 3], 1, padding="SAME", scope="conv1")
    # 64*64*64
    # 第一次池化，大小2*2， 步长为2
    pool1 = slim.max_pool2d(conv1, [2, 2], [2, 2], padding="SAME")
    # 32*32*64

    # 第二层卷积
    conv2 = slim.conv2d(pool1, 128, [3, 3], padding="SAME", scope="conv2")
    # 32*32*128
    # 第二次池化
    pool2 = slim.max_pool2d(conv2, [2, 2], [2, 2], padding="SAME")
    # 16*16*128

    # 第三层卷积
    conv3 = slim.conv2d(pool2, 256, [3, 3], padding="SAME", scope="conv3")
    # 16*16*256
    # 第三次池化
    pool3 = slim.max_pool2d(conv3, [2, 2], [2, 2], padding="SAME")
    # 8*8*256

    # 第四层卷积
    conv4 = slim.conv2d(pool3, 512, [3, 3], [2, 2], scope="conv4", padding="SAME")
    # 4*4*512
    # 第四次池化
    pool4 = slim.max_pool2d(conv4, [2, 2], [2, 2], padding="SAME")
    # 2*2*512

    # 结果转为一维
    flat = slim.flatten(pool4)
    # 2048

    # 两层全连接层
    fcnet1 = slim.fully_connected(slim.dropout(flat, keep_prob=keep_prob), 1024, activation_fn=tf.nn.tanh,
                                  scope="fcnet1")
    # 1024
    fcnet2 = slim.fully_connected(slim.dropout(fcnet1, keep_prob=keep_prob), 3755, activation_fn=None, scope="fcnet2")
    # 3755

    # 损失值
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fcnet2, labels=labels))

    # 计算准确率, argmax按行取最大值的索引
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fcnet2, 1), labels), tf.float32))
    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
    # 指数衰减学习率
    lrate = tf.train.exponential_decay(2e-4, step, decay_rate=0.97, decay_steps=2000, staircase=True)
    # 优化器减少损失值
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss, global_step=step)

    # 结果转为概率值
    prob_dist = tf.nn.softmax(fcnet2)
    # 获取概率top3的概率和及其索引
    val_top3, index_top3 = tf.nn.top_k(prob_dist, 3)

    # top3的准确率
    accuracy_top3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_dist, labels, 3), tf.float32))
    accuracy_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(prob_dist, labels, 5), tf.float32))


    return {
            "conv1": conv1, "pool1": pool1, "conv2": conv2, "pool2": pool2, "conv3": conv3, "pool3": pool3,
            "conv4": conv4, "pool4": pool4, "flat": flat, "fcnet1": fcnet1, "fcnet2": fcnet2, "prob_dist": prob_dist,
            "img": img,
            "label": labels,
            "global_step": step,
            "optimizer": optimizer,
            "loss": loss,
            "accuracy": accuracy,
            'keep_prob': keep_prob,
            "val_top3": val_top3,
            "index_top3": index_top3,
            "accuracy_top3": accuracy_top3,
            "accuracy_top5": accuracy_top5,
            }


def train():
    with tf.Session() as sess:
        print("加载数据......")
        # 获取batch数据
        trn_imgs, trn_labels = batch(FLAGS.train_dir, FLAGS.batch_size, prepocess=True)
        tst_imgs, tst_labels = batch(FLAGS.test_dir, FLAGS.batch_size)
        graph = cnn()
        # 初始化
        sess.run(tf.global_variables_initializer())
        # 设置多线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        saver = tf.train.Saver()

        step = 0
        # 可以从某个保存的模型下接着训练
        if FLAGS.restore:
            # 获取最后一个保存点
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
            if checkpoint:
                # 加载保存点
                saver.restore(sess, checkpoint)
                step += int(checkpoint.split('-')[-1])
                print("从保存点继续训练")
        print("开始训练......")
        while not coord.should_stop():
            # 获取数据
            trn_img_batch, trn_label_batch = sess.run([trn_imgs, trn_labels])
            # 向网络中传入的数据
            graph_dict = {graph['img']: trn_img_batch, graph['label']: trn_label_batch, graph['keep_prob']: 0.8}
            # 传入数据并执行
            opt, loss, step = sess.run(
                [graph['optimizer'], graph['loss'], graph['global_step']], feed_dict=graph_dict)
            print("# " + str(step) + " with loss " + str(loss))
            if step > FLAGS.max_step:
                break
            # 用test数据评估网络
            if (step % 500 == 0) and (step >= 500):
                tst_img_batch, tst_label_batch = sess.run([tst_imgs, tst_labels])
                graph_dict = {graph['img']: tst_img_batch, graph['label']: tst_label_batch, graph['keep_prob']: 1.0}
                accuracy = sess.run([graph['accuracy']], feed_dict=graph_dict)
                print("# " + str(step) + " 准确率: %.8f" % accuracy)
                # 保存
                if step % 10000 == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoint, 'hccr'), global_step=graph['global_step'])
        coord.join(threads)
        saver.save(sess, os.path.join(FLAGS.checkpoint, 'hccr'), global_step=graph['global_step'])
        sess.close()
    return


# 获取汉字label映射表
def get_label_dict():
    f = open('./chinese_labels', 'rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict


# 获待预测图像文件夹内的图像名字
def get_file_list(path):
    list_name = []
    files = os.listdir(path)
    files.sort()
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name


# 测试/预测
def test(path):

    # 读取图片并标准化
    image_list = []
    for each in path:
        # 以灰度模式读取图片
        tst_image = cv2.imread(each, cv2.IMREAD_GRAYSCALE)
        tst_image = cv2.resize(tst_image, (64, 64))
        tst_image = numpy.asarray(tst_image) / 255.0
        tst_image = tst_image.reshape([-1, 64, 64, 1])
        image_list.append(tst_image)

    # 将测试图片传入网络，得到预测值
    with tf.Session() as sess:
        graph = cnn()
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(FLAGS.checkpoint))

        # 概率列表
        val_list = []
        # 索引列表
        index_list = []
        for each_item in image_list:
            test_dict = {graph['img']: each_item, graph['keep_prob']: 1.0}
            val, index = sess.run([graph['val_top3'], graph['index_top3']], feed_dict=test_dict)
            val_list.append(val)
            index_list.append(index)
    return val_list, index_list


def inside():
    each = './inside/inside.png'
    # 读取图片并标准化
    # 以灰度模式读取图片
    tst_image = cv2.imread(each, cv2.IMREAD_GRAYSCALE)
    tst_image = cv2.resize(tst_image, (64, 64))
    tst_image = numpy.asarray(tst_image) / 255.0
    tst_image = tst_image.reshape([-1, 64, 64, 1])

    # 将测试图片传入网络，得到预测值
    with tf.Session() as sess:
        graph = cnn()
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(FLAGS.checkpoint))

        test_dict = {graph['img']: tst_image, graph['label']: [1952], graph['keep_prob']: 1.0}
        # 卷积
        conv1, pool1 = sess.run([graph['conv1'], graph['pool1']], feed_dict=test_dict)
        print(conv1.shape, pool1.shape)
        conv2, pool2 = sess.run([graph['conv2'], graph['pool2']], feed_dict=test_dict)
        print(conv2.shape, pool2.shape)
        conv3, pool3 = sess.run([graph['conv3'], graph['pool3']], feed_dict=test_dict)
        print(conv3.shape, pool3.shape)
        conv4, pool4 = sess.run([graph['conv4'], graph['pool4']], feed_dict=test_dict)
        print(conv4.shape, pool4.shape)
        # 全连接
        flat, fcnet1, fcnet2 = sess.run([graph['flat'], graph['fcnet1'], graph['fcnet2']], feed_dict=test_dict)
        print(flat.shape, fcnet1.shape, fcnet2.shape)
        print(fcnet2)
        # 损失，准确率
        loss = sess.run(graph['loss'], feed_dict=test_dict)
        print(loss)
        accuracy = sess.run(graph['accuracy'], feed_dict=test_dict)
        print(accuracy)
        # 概率
        prob_dist = sess.run(graph['prob_dist'], feed_dict=test_dict)
        print(prob_dist)

        print(sess.run(tf.argmax(fcnet2, 1)))
        print(sess.run(tf.equal(tf.argmax(fcnet2, 1), [1952])))
        print(sess.run(tf.cast(tf.equal(tf.argmax(fcnet2, 1), [1952]), tf.float32)))


def evaluation():
    with tf.Session() as sess:
        print("加载数据......")
        # 获取batch数据
        tst_imgs, tst_labels = batch(FLAGS.test_dir, FLAGS.batch_size)
        graph = cnn()
        # 初始化
        sess.run(tf.global_variables_initializer())
        # 设置多线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        # 加载模型
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        saver.restore(sess, checkpoint)

        all_step = 1

        while not coord.should_stop():
            # 用test数据评估网络
            tst_img_batch, tst_label_batch = sess.run([tst_imgs, tst_labels])
            graph_dict = {graph['img']: tst_img_batch, graph['label']: tst_label_batch, graph['keep_prob']: 1.0}
            accuracy, accuracy_top3, accuracy_top5 = sess.run([graph['accuracy'], graph["accuracy_top3"],
                                                               graph["accuracy_top5"]], feed_dict=graph_dict)
            if all_step > 1200:
                break
            print("step=", all_step, "当前batchsize的准确率: %.8f" % accuracy,
                  "当前batchsize的top3的准确率：%.8f" % accuracy_top3, "当前batchsize的top5的准确率：%.8f" % accuracy_top5)
            all_step += 1
        coord.join(threads)
        sess.close()
    return


def main(_):
    # 训练or测试
    if FLAGS.mode == "train":
        print("训练模型")
        train()
    if FLAGS.mode == "test":
        print("预测")
        # 获取汉字，索性的对照表
        label_dict = get_label_dict()
        # 获取待预测文件夹内的文件名
        name_list = get_file_list(FLAGS.temp)
        final = []
        print("预测结果为（取top3）：")
        predict_val, predict_index = test(name_list)
        for i in range(len(predict_index)):
            print(name_list[i])
            print(
                  label_dict[int(predict_index[i][0][0])], label_dict[int(predict_index[i][0][1])],
                  label_dict[int(predict_index[i][0][2])],
                  "概率值为：", predict_val[i][0][0], predict_val[i][0][1], predict_val[i][0][2],
                  "索引值为：", str(predict_index[i][0][0]), str(predict_index[i][0][1]), str(predict_index[i][0][2]))
            final.append(label_dict[int(predict_index[i][0][0])])
        print("最终预测结果为：", final)
        with open("./test_result_txt/test.txt", 'w') as f:
            f.write(''.join(final))
    if FLAGS.mode == "inside":
        print("内部参数")
        inside()

    if FLAGS.mode == "evaluation":
        print("评估模型")
        evaluation()


if __name__ == '__main__':
    tf.app.run()
