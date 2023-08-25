from typing import List, Dict, Any

import pandas as pd
from keras import Sequential
from keras.callbacks import (EarlyStopping,
                             ReduceLROnPlateau,
                             TensorBoard,
                             ModelCheckpoint)
from keras.layers import Dense, Input
from keras.losses import (MeanSquaredError,
                          MeanAbsoluteError,
                          MeanSquaredLogarithmicError)
from keras.metrics import (MeanAbsoluteError,
                           MeanSquaredError,
                           RootMeanSquaredError,
                           R2Score,
                           MeanSquaredLogarithmicError,
                           MeanAbsolutePercentageError)
from keras.optimizers import (Adam,
                              SGD,
                              RMSprop)
from sklearn.model_selection import train_test_split

from source.dataset_generator import create_shuffled_dataset
from source.furuta_utils import read_yaml_parameters, create_folder_if_not_exists

configuration_path = '../config'
datasets_path = '../data/datasets'
models_path = '../data/models'


def save_dataset(data: pd.DataFrame,
                 folder_to_save: str,
                 file_name: str) -> None:

    create_folder_if_not_exists(folder_path=folder_to_save)
    saving_path = f'{folder_to_save}/{file_name}.parquet'
    data.to_parquet(path=saving_path,
                    index=False)


def save_model(model,
               folder_to_save,
               name):

    create_folder_if_not_exists(folder_path=folder_to_save)
    model_path = f'{folder_to_save}/{name}.h5'
    model.save(model_path)


def create_model_architecture(input_shape,
                              number_units,
                              activation_hidden_layers,
                              activation_output_layer) -> Sequential:
    model = Sequential()

    model.add(Input(shape=(input_shape, )))

    number_layers = len(number_units)
    last_layer_index = number_layers - 1

    for layer_index, layer_units in enumerate(number_units):
        activation_function = activation_hidden_layers
        if layer_index == last_layer_index:
            activation_function = activation_output_layer
        model.add(Dense(units=layer_units,
                        activation=activation_function))

    return model


def get_optimizer(optimizer_type: str,
                  learning_rate: float) -> "keras.optimizers":

    optimizer_classes = {'adam': Adam,
                         'sgd': SGD,
                         'rmsprop': RMSprop}

    try:
        selected_optimizer = optimizer_classes[optimizer_type](learning_rate=learning_rate)
        return selected_optimizer
    except KeyError:
        print(f'Optimizer {optimizer_type} not implemented yet.')


def get_loss_function(loss_type: List[str]) -> "keras.losses":
    loss_type, = loss_type
    loss_function_classes = {'mse': MeanSquaredError,
                             'mae': MeanAbsoluteError,
                             'msle': MeanSquaredLogarithmicError}

    try:
        selected_loss_function = loss_function_classes[loss_type]()
        return selected_loss_function
    except KeyError:
        print(f'Loss function {loss_type} not implemented yet.')


def get_metrics(metrics_type: List[str]) -> List["keras.metrics"]:
    metrics_classes = {'mae': MeanAbsoluteError,
                       'mse': MeanSquaredError,
                       'rmse': RootMeanSquaredError,
                       'mape': MeanAbsolutePercentageError,
                       'msle': MeanSquaredLogarithmicError,
                       'r2': R2Score}

    try:
        selected_metrics = [metrics_classes[metric]() for metric in metrics_type]
        return selected_metrics
    except KeyError:
        print(f'Metrics {metrics_type} not implemented yet.')


def get_callbacks(callback_type: str, configuration: Dict[str, Any]):
    callbacks_classes = {'early_stopping': EarlyStopping,
                         'model_checkpoint': ModelCheckpoint,
                         'reduce_lr': ReduceLROnPlateau,
                         'tensor_board': TensorBoard}
    try:
        selected_callback = callbacks_classes[callback_type](**configuration)
        return selected_callback
    except KeyError:
        print(f'Callback {callback_type} not implemented yet')


if __name__ == '__main__':
    global_configuration = read_yaml_parameters(folder_path=configuration_path)

    # SET DIFFERENT CONFIGURATIONS
    matlab_configuration_training = global_configuration['matlab_data']['training']
    dataset_configuration_training = global_configuration['datasets']['training']

    matlab_configuration_dev_test = global_configuration['matlab_data']['validation_and_test']
    dataset_configuration_dev_test = global_configuration['datasets']['validation_and_test']

    neural_network_configuration = global_configuration['model']
    callbacks_configuration = global_configuration['callbacks']
    training_configuration = global_configuration['training_model']

    # CREATE OR LOAD TRAINING DATASET
    if dataset_configuration_training['create_dataset']:
        training_dataset = create_shuffled_dataset(matlab_configuration=matlab_configuration_training,
                                                   dataset_configuration=dataset_configuration_training)

        if dataset_configuration_training['save_dataset']:
            save_dataset(data=training_dataset,
                         folder_to_save=datasets_path,
                         file_name=dataset_configuration_training['name'])
    else:
        training_dataset = pd.read_parquet(path=f"{datasets_path}/{dataset_configuration_training['name']}")

    # CREATE OR LOAD DEV AND TEST DATASET
    if dataset_configuration_dev_test['create_dataset']:
        dev_test_dataset = create_shuffled_dataset(matlab_configuration=matlab_configuration_dev_test,
                                                   dataset_configuration=dataset_configuration_dev_test)

        if dataset_configuration_dev_test['save_dataset']:
            save_dataset(data=dev_test_dataset,
                         folder_to_save=datasets_path,
                         file_name=dataset_configuration_dev_test['name'])
    else:
        dev_test_dataset = pd.read_parquet(path=f"{datasets_path}/{dataset_configuration_dev_test['name']}")

    # CREATE THE ARCHITECTURE
    furuta_pendulum_model = create_model_architecture(input_shape=neural_network_configuration['architecture']['input_shape'],
                                                      number_units=neural_network_configuration['architecture']['number_units'],
                                                      activation_hidden_layers=neural_network_configuration['architecture']['activation_hidden_layers'],
                                                      activation_output_layer=neural_network_configuration['architecture']['activation_output_layer'])

    # SET OPTIMIZER, LOSS FUNCTION, METRICS AND CALLBACKS
    optimizer = get_optimizer(optimizer_type=neural_network_configuration['optimizer']['optimizer_type'],
                              learning_rate=neural_network_configuration['optimizer']['learning_rate'])
    loss = get_loss_function(loss_type=neural_network_configuration['loss'])
    metrics = get_metrics(metrics_type=neural_network_configuration['metrics'])
    callbacks = [get_callbacks(callback_type, configuration) for callback_type, configuration in callbacks_configuration.items()]

    furuta_pendulum_model.compile(loss=loss,
                                  optimizer=optimizer,
                                  metrics=metrics)

    # LOAD MODEL IF WE WANT TO RE-TRAIN
    if training_configuration['retrain']:
        furuta_pendulum_model.load_weights(callbacks_configuration['model_checkpoint']['filepath'])

    # SEPARATE FEATURES AND TARGETS FROM TRAINING, DEV AND TEST SETS
    train_features, train_target = training_dataset[training_configuration['features']], training_dataset[training_configuration['target']]
    dev_test_features, dev_test_targets = dev_test_dataset[training_configuration['features']], dev_test_dataset[training_configuration['target']]

    dev_features, test_features, dev_target, test_target = train_test_split(dev_test_features.values,
                                                                            dev_test_targets.values,
                                                                            test_size=training_configuration['test_size'],
                                                                            shuffle=False)
    # TRAIN THE NEURAL NETWORK
    history = furuta_pendulum_model.fit(train_features.values,
                                        train_target.values,
                                        epochs=training_configuration['epochs'],
                                        batch_size=training_configuration['batch_size'],
                                        validation_data=(dev_features.values, dev_target.values),
                                        callbacks=callbacks,
                                        verbose=1,
                                        shuffle=False)

    # SAVE MODEL IN DATA FOLDER
    save_model(model=furuta_pendulum_model,
               folder_to_save=models_path,
               name=training_configuration['model_name'])

    scores = furuta_pendulum_model.evaluate(test_features.values,
                                            test_target.values)
    predictions = furuta_pendulum_model.predict(test_features)
