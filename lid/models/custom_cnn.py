def build_model(bands=32, frames=31, channels=1, n_classes=3, **kwargs):
    from keras import regularizers
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
    from keras.layers import Dropout, Activation
    from keras.models import Sequential
    from keras.optimizers import SGD

    model = Sequential()
    input_shape = (bands, frames, channels)
    # 40x1000

    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 20x500

    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 10x250
    model.add(Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 5x125
    model.add(Conv2D(128, (3, 5), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3, 5), strides=(1, 5), padding='same'))

    # 5x25
    model.add(Conv2D(256, (3, 4), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('elu'))
    # model.add(MaxPooling2D(pool_size=(3, 2), strides=(1, 2), padding='same'))
    model.add(AveragePooling2D(pool_size=(3, 1), strides=(1, 2), padding='valid'))

    # 1x1

    model.add(Flatten())

    model.add(Dense(32, activation='elu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(Dropout(0.5))

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=False)
    #
    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=sgd,
    #     metrics=['accuracy'])

    return model


def main():
    m = build_model()
    m.summary()


if __name__ == '__main__':
    main()
