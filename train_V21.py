# -*- coding:utf-8 -*-
from random import shuffle, random
from model_V21 import *

import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_height": 128,
                           
                           "img_width": 88,

                           "lr": 0.0002,

                           "batch_size": 100,

                           "epochs": 200,
                           
                           "tr_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/train.txt",
                           
                           "tr_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_train.txt",
                           
                           "tr_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/motion_blur2/",

                           "re_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/train.txt",
                           
                           "re_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_train.txt",
                           
                           "re_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI/",
                           
                           "te_txt_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/test.txt",
                           
                           "te_txt_name": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/GEI_IDList_test_fix.txt",
                           
                           "te_img_path": "D:/[1]DB/[4]etc_experiment/Body_age/OULP-Age/motion_blur2/",

                           "train": True,

                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",

                           "save_checkpoint": "/content/drive/MyDrive/4th_paper/GEI_age_estimation/V21_checkpoint",
                           
                           "graphs": "/content/drive/MyDrive/4th_paper/GEI_age_estimation/"})

optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def train_input(tr_data, tr_label):

    tr_img = tf.io.read_file(tr_data[0])
    tr_img = tf.image.decode_png(tr_img, 1)
    tr_img = tf.image.resize(tr_img, [FLAGS.img_height, FLAGS.img_width]) / 127.5 - 1.
    
    re_img = tf.io.read_file(tr_data[1])
    re_img = tf.image.decode_png(re_img, 1)
    re_img = tf.image.resize(re_img, [FLAGS.img_height, FLAGS.img_width]) / 127.5 - 1.

    tr_lab = tr_label[0]
    re_lab = tr_label[1]

    return tr_img, tr_lab, re_img, re_lab

def test_input(te_data, te_label):

    te_img = tf.io.read_file(te_data)
    te_img = tf.image.decode_png(te_img, 1)
    te_img = tf.image.resize(te_img, [FLAGS.img_height, FLAGS.img_width]) / 127.5 - 1.

    te_lab = te_label

    return te_img, te_lab

#@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_CDF(logits, labels, num_feature):

    logit_CDF = []
    label_CDF = []
    for i in range(FLAGS.batch_size):
        log = 0.
        lab = 0.
        logit = logits[i]
        logit = logit.numpy()
        label = labels[i]
        label = label.numpy()
        log_buf = []
        lab_buf = []
        for j in range(num_feature):
            log += logit[j]
            lab += label[j]
            log_buf.append(log)
            lab_buf.append(lab)

        logit_CDF.append(log_buf)
        label_CDF.append(lab_buf)

    return logit_CDF, label_CDF

def make_label_V2(ori_label):
    l = []
    for i in range(FLAGS.batch_size):
        label = [1] * (ori_label[i].numpy() + 1) + [0] * (88 - (ori_label[i].numpy() + 1))
        label = tf.cast(label, tf.float32)
        l.append(label)
    return tf.convert_to_tensor(l, tf.float32)

def cal_loss(motion_GAN, original_GAN, motion_regression_model, original_regression_model,discrim_motion,discrim_original,
             batch_tr_img, batch_tr_lab, batch_re_img, batch_re_lab, batch_rank_labels):

    with tf.GradientTape(persistent=True) as tape:
        # batch_re_img --> ground truth image
        fake_motion_img = run_model(motion_GAN, batch_tr_img, True)
        fake_original_img = run_model(original_GAN, batch_re_img, True)

        motion_logit = run_model(motion_regression_model, fake_motion_img, True)
        original_logit = run_model(original_regression_model, fake_original_img, True)

        true_original_d = run_model(discrim_original, batch_re_img, True)
        fake_original_d = run_model(discrim_original, fake_original_img, True)
        fake_motion_d = run_model(discrim_motion, fake_motion_img, True)

        adv_loss = (tf.reduce_mean((true_original_d - tf.ones_like(true_original_d))**2) + tf.reduce_mean((fake_original_d - tf.zeros_like(fake_original_d))**2)) / 2 \
            + tf.reduce_mean((fake_motion_d - tf.zeros_like(fake_motion_d))**2) / 2
        #print(adv_loss)
        pixel_loss = tf.reduce_mean(tf.abs(fake_motion_img - batch_re_img)) + tf.reduce_mean(tf.abs(fake_original_img - batch_re_img))
        #print(pixel_loss)
        age_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(batch_rank_labels, motion_logit) \
            + tf.keras.losses.BinaryCrossentropy(from_logits=True)(batch_rank_labels, original_logit) \
            + tf.keras.losses.MeanAbsoluteError()(motion_logit, original_logit)
        #age_loss = tf.keras.losses.MeanAbsoluteError()(batch_re_lab, motion_logit) + tf.keras.losses.MeanAbsoluteError()(batch_re_lab, original_logit) \
        #    + tf.keras.losses.MeanAbsoluteError()(motion_logit, original_logit)

        total_loss = 0.0001 * adv_loss + pixel_loss + 10. * age_loss

    gener_grads = tape.gradient(total_loss, motion_GAN.trainable_variables + original_GAN.trainable_variables)
    discrim_grads = tape.gradient(total_loss, discrim_original.trainable_variables + discrim_motion.trainable_variables)
    regress_grads = tape.gradient(total_loss, motion_regression_model.trainable_variables + original_regression_model.trainable_variables)

    optim.apply_gradients(zip(gener_grads, motion_GAN.trainable_variables + original_GAN.trainable_variables))
    optim.apply_gradients(zip(discrim_grads, discrim_original.trainable_variables + discrim_motion.trainable_variables))
    optim.apply_gradients(zip(regress_grads, motion_regression_model.trainable_variables + original_regression_model.trainable_variables))

    return total_loss

def cal_mae(motion_GAN, motion_regression_model, images, labels):

    fake_motion_img = run_model(motion_GAN, images, False)
    motion_logit = run_model(motion_regression_model, fake_motion_img, False)

    ae = 0
    for i in range(137):

        age_predict = tf.nn.sigmoid(motion_logit[i])
        age_predict = tf.cast(tf.less(0.5, age_predict), tf.int32)
        age_predict = tf.reduce_sum(age_predict)

        ae += tf.abs(labels[i] - age_predict)

    return ae

def main():
    motion_GAN = GL_GAN(input_shape=(FLAGS.img_height, FLAGS.img_width, 1))
    original_GAN = GL_GAN(input_shape=(FLAGS.img_height, FLAGS.img_width, 1))
    motion_regression_model = regression_model(input_shape=(FLAGS.img_height, FLAGS.img_width, 1))
    original_regression_model = regression_model(input_shape=(FLAGS.img_height, FLAGS.img_width, 1))
    discrim_motion = discriminator(input_shape=(FLAGS.img_height, FLAGS.img_width, 1))
    discrim_original = discriminator(input_shape=(FLAGS.img_height, FLAGS.img_width, 1))

    motion_GAN.summary()
    original_GAN.summary()
    motion_regression_model.summary()
    original_regression_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(motion_GAN=motion_GAN,
                                   original_GAN=original_GAN,
                                   motion_regression_model=motion_regression_model,
                                   original_regression_model=original_regression_model,
                                   discrim_motion=discrim_motion,
                                   discrim_original=discrim_original,
                                   optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!!")

    if FLAGS.train:
        count = 0

        tr_img = np.loadtxt(FLAGS.tr_txt_name, dtype="<U100", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img + ".png"for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        re_img = np.loadtxt(FLAGS.re_txt_name, dtype="<U100", skiprows=0, usecols=0)
        re_img = [FLAGS.re_img_path + img + ".png"for img in re_img]
        re_lab = np.loadtxt(FLAGS.re_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_img = np.loadtxt(FLAGS.te_txt_name, dtype="<U100", skiprows=0, usecols=0)
        te_img = [FLAGS.te_img_path + img + ".png" for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(test_input)
        te_gener = te_gener.batch(137)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        #############################
        # Define the graphs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + "_V21" + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        val_log_dir = FLAGS.graphs + current_time + "_V21" + '/val'
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        #############################

        for epoch in range(FLAGS.epochs):

            A = list(zip(tr_img, tr_lab))
            shuffle(A)
            tr_img, tr_lab = zip(*A)

            B = list(zip(re_img, re_lab))
            shuffle(B)
            re_img, re_lab = zip(*B)

            tr_data = list(zip(tr_img, re_img))
            tr_label = list(zip(tr_lab, re_lab))

            tr_data, tr_label = np.array(tr_data), np.array(tr_label)

            gener = tf.data.Dataset.from_tensor_slices((tr_data, tr_label))
            gener = gener.map(train_input)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(gener)
            tr_idx = len(tr_img) // FLAGS.batch_size

            for step in range(tr_idx):

                batch_tr_img, batch_tr_lab, batch_re_img, batch_re_lab = next(tr_iter)
                batch_rank_labels = make_label_V2(batch_tr_lab)

                loss = cal_loss(motion_GAN, original_GAN, motion_regression_model, original_regression_model,discrim_motion,discrim_original,
                                batch_tr_img, batch_tr_lab, batch_re_img, batch_re_lab, batch_rank_labels)

                with train_summary_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=count)

                if count % 10 == 0:
                    print("Epochs = {} [{}/{}] Loss = {}".format(epoch, step + 1, tr_idx, loss))

                if count % 100 == 0:
                    te_iter = iter(te_gener)
                    te_idx = len(te_img) // 137
                    ae = 0
                    for i in range(te_idx):

                        test_images, test_labels = next(te_iter)

                        ae = cal_mae(motion_GAN, motion_regression_model, test_images, test_labels)
                    MAE = ae / len(te_img)
                    print("================================")
                    print("step = {}, MAE = {}".format(count, MAE))
                    print("================================")
                    with val_summary_writer.as_default():
                        tf.summary.scalar('MAE', MAE, step=count)

                    num_ = int(count // 100)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                    if not os.path.isdir(model_dir):
                       os.makedirs(model_dir)
                       print("Make {} files to save checkpoint".format(num_))

                    ckpt = tf.train.Checkpoint(motion_GAN=motion_GAN,
                                               original_GAN=original_GAN,
                                               motion_regression_model=motion_regression_model,
                                               original_regression_model=original_regression_model,
                                               discrim_motion=discrim_motion,
                                               discrim_original=discrim_original,
                                               optim=optim)
                    ckpt_dir = model_dir + "/" + "V21_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)

                count += 1


if __name__ == "__main__":
    main()