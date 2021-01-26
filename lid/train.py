import argparse
import datetime
import functools
import json
import math
import os.path

import keras
import numpy as np
from sklearn.metrics import accuracy_score

from lid import common, models
from lid.datautil import load_training_data
from lid.featurization import AudioFeature
from lid.trainutils import (batch_generator, history_dataframe, setup_keras, LogCallback, predict_voted)

np.random.seed(137)


def train_model(out_dir, train_df, val_df, model, sample_featurizer, settings):
    epochs = settings['epochs']
    batch_size = settings['batch']
    learning_rate = settings['learning_rate']

    optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=settings['nesterov_momentum'], nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Prepare data loaders for train and val items
    def load_f(sample_fpath, validation, start_time=None):
        do_augment = not validation and settings['augment']
        return sample_featurizer.load_sample(sample_fpath, window_frames=settings['frames'], augment=do_augment,
                                             start_time=start_time, normalize=settings['normalize'])

    train_loader = functools.partial(load_f, validation=False)
    val_loader = functools.partial(load_f, validation=True)

    # Prepare callbacks
    model_path = os.path.join(out_dir, r'e{epoch:02d}-v{val_accuracy:.2f}.t{loss:.2f}.hdf5')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', verbose=True, save_best_only=True)

    def voted_score():
        vote_df = val_df.groupby('Y', as_index=False).apply(lambda x: x.sample(settings['voting_test_count'] // 3)).reset_index(drop=True)
        y_pred_mean, y_pred_majority = predict_voted(model, vote_df, loader=val_loader, sample_featurizer=sample_featurizer,
                                                     window_frames=settings['frames'], method=settings['voting'], overlap=settings['voting_overlap'],
                                                     seconds_limit=settings['voted_seconds'])
        acc_mean = accuracy_score(vote_df.Y, np.argmax(y_pred_mean, axis=1))
        acc_majority = accuracy_score(vote_df.Y, np.argmax(y_pred_majority, axis=1))
        d = {
            'voted_mean_acc': acc_mean,
            'voted_majority_acc': acc_majority
        }
        for k, v in d.items():
            print("{}: {:.4f}".format(k, v))
        return d

    log_path = os.path.join(out_dir, 'train.csv')
    log_callback = LogCallback(log_path, voted_score)
    callbacks_list = [checkpoint_callback, log_callback]

    # Prepare batch generators
    train_gen = batch_generator(train_df.fullpath, train_df.Y, loader=train_loader, batchsize=batch_size)
    val_gen = batch_generator(val_df.fullpath, val_df.Y, loader=val_loader, batchsize=batch_size)

    # dump_validation_data(val_gen)  # check availability of validation data for callbacks.ModelCheckpoint

    # Train model
    common.ensure_dir(out_dir)

    hist = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=math.ceil(train_df.shape[0] / batch_size),
                     validation_steps=math.ceil(val_df.shape[0] / batch_size), callbacks=callbacks_list, epochs=epochs,
                     verbose=True, workers=3, use_multiprocessing=True)

    df = history_dataframe(hist)
    history_path = os.path.join(out_dir, 'history.csv')
    df.to_csv(history_path)

    return hist


def parse():
    parser = argparse.ArgumentParser(description='Train a model')

    common.add_arguments(parser)
    a = parser.add_argument

    a('--load_name', type=str, default='', help='Load a already trained model')
    return parser.parse_args()


def main():
    """
    parse -> load settings -> train_model
    """
    setup_keras()

    args = parse()

    train_settings = common.load_settings(args.settings_path, default_conf_name='train.yml')
    train_settings['store'] = args.store

    feature_settings = common.load_settings(args.settings_path, default_conf_name='feature.yml')
    model_settings = common.load_settings(args.settings_path, default_conf_name=train_settings['model_conf'])

    train_df, val_df = load_training_data(dict(train_settings, **feature_settings))
    assert train_df.shape[0] > val_df.shape[0] * 4.5, f'training data {train_df.shape[0]} should be much larger than validation {val_df.shape[0]}'

    sample_featurizer = AudioFeature(feature_settings)

    if args.load_name:
        model_name = args.load_name
        print('Loading existing model', model_name)
        m = keras.models.load_model(model_name)
    else:
        t = datetime.datetime.now().strftime('%Y%m%d-%H%M')
        model_name = f"model-{model_settings['model']}_hop{feature_settings['hop_length']}_{t}"
        m = models.build(dict(model_settings, **feature_settings))
    m.summary()

    output_dir = os.path.join(args.model_store, model_name)

    print(f"Training model: '{model_name}'", json.dumps(train_settings, indent=1))

    combined_settings = dict(train_settings, **model_settings, **feature_settings)

    h = train_model(output_dir, train_df, val_df,
                    model=m,
                    sample_featurizer=sample_featurizer,
                    settings=combined_settings)


if __name__ == '__main__':
    main()
