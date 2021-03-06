from keras.layers import Convolution2D, MaxPooling2D, SeparableConv2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2


def build_model(frames=128, bands=128, channels=1, num_labels=3,
                conv_size=(5, 5), conv_block='conv',
                downsample_size=(4, 2),
                fully_connected=64,
                n_stages=None, n_blocks_per_stage=None,
                filters=24, kernels_growth=2,
                dropout=0.5,
                use_strides=False):
    """
    Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification, 2016.
    https://arxiv.org/pdf/1608.04363.pdf

    Based on https://gist.github.com/jaron/5b17c9f37f351780744aefc74f93d3ae
    """
    assert conv_block in ('conv', 'depthwise_separable')
    Conv2 = SeparableConv2D if conv_block == 'depthwise_separable' else Convolution2D
    kernel = conv_size
    if use_strides:
        strides = downsample_size
        pool = (1, 1)
    else:
        strides = (1, 1)
        pool = downsample_size
    block1 = [
        Convolution2D(filters, kernel, padding='same', strides=strides, input_shape=(bands, frames, channels)),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool, strides=(2, 2)),
        Activation('relu'),
    ]
    block2 = [
        Conv2(filters * kernels_growth, kernel, padding='same', strides=strides),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool, strides=(2, 2)),
        Activation('relu'),
    ]
    block3 = [
        Conv2(filters * kernels_growth, kernel, padding='valid', strides=strides),
        BatchNormalization(),
        Activation('relu'),
    ]
    dnn = [
        Flatten(),

        Dropout(dropout),
        Dense(fully_connected * 2, kernel_regularizer=l2(0.001)),
        Activation('relu'),

        Dropout(dropout),
        Dense(fully_connected, kernel_regularizer=l2(0.001)),
        Activation('relu'),

        Dropout(dropout),
        Dense(num_labels, kernel_regularizer=l2(0.001)),
        Activation('softmax'),
    ]
    layers = block1 + block2 + block3 + dnn
    # layers = block1 + block2 + dnn
    model = Sequential(layers)
    return model
