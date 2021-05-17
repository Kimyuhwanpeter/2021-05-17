# -*- codingL:utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2

def GL_GAN(input_shape=(128, 88, 1), weight_decay=0.00001):

    h = inputs = tf.keras.Input(input_shape)

    crop_1 = tf.image.crop_to_bounding_box(h, 0, 0, 22, 88)
    crop_2 = tf.image.crop_to_bounding_box(h, 22, 0, 48, 88)
    crop_3 = tf.image.crop_to_bounding_box(h, 70, 0, 58, 88)

    ###########################################################################################
    crop_1 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_1)
    crop_1 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)

    crop_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_1)

    crop_1 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)
        
    ###########################################################################################

    ###########################################################################################
    crop_2 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_2)
    crop_2 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)

    crop_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_2)

    crop_2 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)
    ###########################################################################################

    ###########################################################################################
    crop_3 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    crop_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_3)

    crop_3 = tf.keras.layers.ZeroPadding2D((1, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=2,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    ###########################################################################################
    
    crop = tf.concat([crop_1, crop_2, crop_3], 1)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.concat([h, crop], -1)

    # decoder part

    h = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=7,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2DTranspose(filters=32,
                                        kernel_size=7,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(filters=1,
                               kernel_size=7,
                               strides=1,
                               padding="same")(h)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def regression_model(input_shape=(128, 88, 1), weight_decay=0.00001):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)
    h = tf.nn.local_response_normalization(h)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=5,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)
    h = tf.nn.local_response_normalization(h)

    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)
    h = tf.nn.local_response_normalization(h)

    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)
    h = tf.nn.local_response_normalization(h)

    #h_in = h
    ##################################################################################################
    h_position_att_B = tf.keras.layers.Conv2D(filters=256 // 8,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=False,
                                              kernel_regularizer=l2(weight_decay))(h)   # query 영향을 받는 feature
    h_position_att_B =  tf.keras.layers.Reshape((8*6, 256 // 8))(h_position_att_B)

    h_position_att_C = tf.keras.layers.Conv2D(filters=256 // 8,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=False,
                                              kernel_regularizer=l2(weight_decay))(h)   # key 영향을 주는 feature
    h_position_att_C = tf.keras.layers.Reshape((256 // 8, 8*6))(h_position_att_C)

    e = tf.matmul(h_position_att_B, h_position_att_C)   # 유사도 계산    (query와 key의 내적은 유사도를 측정하는것과 같다)
    attention = tf.nn.softmax(e, -1)
    attention = tf.reshape(attention, [-1, attention.shape[2], attention.shape[1]])


    h_position_att_D = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=1,
                                              strides=1,
                                              use_bias=False,
                                              kernel_regularizer=l2(weight_decay))(h)   # value 이 영향들에 대한 가중치
    h_position_att_D = tf.keras.layers.Reshape((256, 8*6))(h_position_att_D)
    h_position_out = tf.matmul(h_position_att_D, attention)
    h_position_out = tf.keras.layers.Reshape((8, 6, 256))(h_position_out)
    h_position_out = h_position_out + h
    ##################################################################################################

    ##################################################################################################
    h_channel_att_A = tf.keras.layers.Reshape((256, 8*6))(h)    # query
    h_channel_att_B = tf.keras.layers.Reshape((8*6, 256))(h)    # key
    e2 = tf.matmul(h_channel_att_A, h_channel_att_B)    # query 와 key의 유사도 계산
    e2 = tf.reduce_max(e2, -1, keepdims=True) - e2
    attention2 = tf.nn.softmax(e2, -1)

    value = tf.keras.layers.Reshape((256, 8*6))(h)

    h_channel_out = tf.matmul(attention2, value)
    h_channel_out = tf.keras.layers.Reshape((8, 6, 256))(h_channel_out)
    h_channel_out = h_channel_out + h
    ##################################################################################################

    h = h_channel_out + h_position_out  # [8. 6, 128]
    #h = tf.multiply(h, tf.nn.softmax(h_in, -1))    # 이렇게하면 어떤효과가 있을지? --> 각 vector에 해당하는 좌료 값들에 대한 컨트라스트

    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(88)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def discriminator(input_shape=(128, 88, 1), weight_decay=0.00001):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=5,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=5,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.LeakyReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=5,
                               strides=2,
                               padding="same")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)