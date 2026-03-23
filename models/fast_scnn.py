import tensorflow as tf


def conv_bn_relu(x, filters, kernel_size=3, strides=1):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def dsconv(x, filters, strides=1):
    x = tf.keras.layers.DepthwiseConv2D(
        3, strides=strides, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters, 1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def bottleneck_block(x, out_channels, expansion=6, strides=1):
    in_channels = int(x.shape[-1])
    shortcut = x

    y = tf.keras.layers.Conv2D(
        in_channels * expansion, 1, padding="same", use_bias=False
    )(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)

    y = tf.keras.layers.DepthwiseConv2D(
        3, strides=strides, padding="same", use_bias=False
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)

    y = tf.keras.layers.Conv2D(
        out_channels, 1, padding="same", use_bias=False
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)

    if strides == 1 and in_channels == out_channels:
        y = tf.keras.layers.Add()([shortcut, y])

    return y


def pyramid_pooling_block(x, bins=(1, 2, 4)):
    h = int(x.shape[1])
    w = int(x.shape[2])

    concat_list = [x]

    for b in bins:
        pool_h = max(1, h // b)
        pool_w = max(1, w // b)

        pooled = tf.keras.layers.AveragePooling2D(
            pool_size=(pool_h, pool_w),
            strides=(pool_h, pool_w),
            padding="same"
        )(x)

        pooled = conv_bn_relu(pooled, 32, kernel_size=1, strides=1)
        pooled = tf.keras.layers.Resizing(h, w, interpolation="bilinear")(pooled)
        concat_list.append(pooled)

    x = tf.keras.layers.Concatenate()(concat_list)
    x = conv_bn_relu(x, 128, kernel_size=1, strides=1)
    return x


def feature_fusion_block(high_res, low_res, out_channels):
    h = int(high_res.shape[1])
    w = int(high_res.shape[2])

    low_res_up = tf.keras.layers.Resizing(h, w, interpolation="bilinear")(low_res)
    low_res_up = tf.keras.layers.Conv2D(
        out_channels, 1, padding="same", use_bias=False
    )(low_res_up)
    low_res_up = tf.keras.layers.BatchNormalization()(low_res_up)

    high_res_proj = tf.keras.layers.Conv2D(
        out_channels, 1, padding="same", use_bias=False
    )(high_res)
    high_res_proj = tf.keras.layers.BatchNormalization()(high_res_proj)

    x = tf.keras.layers.Add()([high_res_proj, low_res_up])
    x = tf.keras.layers.ReLU()(x)
    return x


def build_fast_scnn(input_shape=(256, 256, 3), num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)

    x = conv_bn_relu(inputs, 32, 3, strides=2)
    x = dsconv(x, 48, strides=2)
    x = dsconv(x, 64, strides=2)
    high_res = x

    x = bottleneck_block(x, 64, expansion=6, strides=2)
    x = bottleneck_block(x, 64, expansion=6, strides=1)
    x = bottleneck_block(x, 96, expansion=6, strides=2)
    x = bottleneck_block(x, 96, expansion=6, strides=1)
    x = bottleneck_block(x, 128, expansion=6, strides=1)

    x = pyramid_pooling_block(x)
    x = feature_fusion_block(high_res, x, out_channels=128)

    x = dsconv(x, 128, strides=1)
    x = dsconv(x, 128, strides=1)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv2D(
        num_classes, 1, padding="same", activation="sigmoid"
    )(x)

    outputs = tf.keras.layers.Resizing(
        input_shape[0], input_shape[1], interpolation="bilinear"
    )(x)

    return tf.keras.Model(inputs, outputs, name="FastSCNN")
