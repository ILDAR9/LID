from keras.layers import Input, Dense, LSTM, Reshape, Dropout
from keras.models import Model
from keras.regularizers import l2

def build_model(bands=32, frames=31, channels=1, n_classes=3, dropout=0.25, fully_connected=100, reccurent_droupdout=0.3, **kwargs):
    input_shape = (bands, frames, channels)

    input = Input(input_shape, name='rnn_input')

    reshaped_input = Reshape((frames, bands))(input)

    layer1 = LSTM(frames, return_sequences=True, name='rnn1')(reshaped_input)
    layer2 = LSTM(frames, return_sequences=False, name='rnn2')(layer1)
    layer3 = Dense(fully_connected, activation='relu', name='dnn3', kernel_regularizer=l2(0.001))(layer2)
    print('dropout', dropout)
    layer3_drop = Dropout(dropout)(layer3)
    rnn_output = Dense(n_classes, activation='softmax', name='rnn_output')(layer3_drop)
    model = Model(inputs=input, outputs=rnn_output)

    return model


def main():
    m = build_model()

    m.summary()


if __name__ == '__main__':
    main()
