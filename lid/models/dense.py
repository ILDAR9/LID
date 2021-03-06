
def build_model(bands=60, frames=41, channels=1, n_labels=3,
                dropout=0.0, depth=7, block=2, growth=15, pooling='avg',
                bottleneck=False, reduction=0.0, subsample=True):
    """
    DenseNet
    """

    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/densenet.py
    from keras_contrib.applications import densenet
    
    input_shape = (bands, frames, channels)

    model = densenet.DenseNet(input_shape=input_shape, pooling=pooling,
                                depth=depth, nb_dense_block=block, growth_rate=growth,
                                bottleneck=bottleneck, reduction=reduction,
                                subsample_initial_block=subsample,
                                include_top=True, classes=n_labels, dropout_rate=dropout)

    return model

def main():
    m = build_model()
    m.save('densenet.hdf5')

    m.summary()

if __name__ == '__main__':
    main()
