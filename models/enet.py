import tensorflow as tf


def conv_bn_prelu(x, filters, kernel_size=3, strides=1):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x


def initial_block(inputs, out_channels=16):
    conv = tf.keras.layers.Conv2D(
        out_channels - 3,
        3,
        strides=2,
        padding="same",
        use_bias=False
    )(inputs)
    pool = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([conv, pool])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x


def bottleneck(x, out_channels, downsample=False, dropout_rate=0.1):
    in_channels = x.shape[-1]
    stride = 2 if downsample else 1

    main = x
    if downsample:
        main = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(main)
        ch_diff = out_channels - int(main.shape[-1])
        if ch_diff > 0:
            main = tf.keras.layers.Lambda(
                lambda t: tf.pad(t, [[0, 0], [0, 0], [0, 0], [0, ch_diff]])
            )(main)

    y = tf.keras.layers.Conv2D(in_channels // 4, 1, padding="same", use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.PReLU(shared_axes=[1, 2])(y)

    y = tf.keras.layers.Conv2D(
        in_channels // 4,
        3,
        strides=stride,
        padding="same",
        use_bias=False
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.PReLU(shared_axes=[1, 2])(y)

    y = tf.keras.layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dropout(dropout_rate)(y)

    x = tf.keras.layers.Add()([main, y])
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x


def up_bottleneck(x, out_channels):
    y = tf.keras.layers.Conv2DTranspose(
        out_channels,
        3,
        strides=2,
        padding="same",
        use_bias=False
    )(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.PReLU(shared_axes=[1, 2])(y)
    return y


def build_enet(input_shape=(256, 256, 3), num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)

    x = initial_block(inputs, out_channels=16)

    x = bottleneck(x, 64, downsample=True, dropout_rate=0.01)
    for _ in range(4):
        x = bottleneck(x, 64, downsample=False, dropout_rate=0.01)

    x = bottleneck(x, 128, downsample=True, dropout_rate=0.1)
    for _ in range(2):
        x = bottleneck(x, 128, downsample=False, dropout_rate=0.1)

    x = up_bottleneck(x, 64)
    x = bottleneck(x, 64, downsample=False, dropout_rate=0.01)

    x = up_bottleneck(x, 16)
    x = bottleneck(x, 16, downsample=False, dropout_rate=0.01)

    x = tf.keras.layers.Conv2DTranspose(
        num_classes,
        2,
        strides=2,
        padding="same",
        activation="sigmoid"
    )(x)

    return tf.keras.Model(inputs, x, name="ENet")
