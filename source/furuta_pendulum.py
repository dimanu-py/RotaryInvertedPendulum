import pandas as pd
from keras import Sequential
from keras.layers import Dense, Input
from keras.optimizers import (Adam,
                              SGD,
                              RMSprop)
from keras.losses import (MeanSquaredError,
                          MeanAbsoluteError,
                          MeanSquaredLogarithmicError)
from keras.metrics import (MeanAbsoluteError,
                           MeanSquaredError,
                           RootMeanSquaredError,
                           R2Score,
                           MeanSquaredLogarithmicError,
                           MeanAbsolutePercentageError)
from typing import List

from source.dataset_generator import generate_datasets
from source.furuta_utils import read_yaml_parameters, create_folder_if_not_exists

configuration_path = '../config'
datasets_path = '../data/datasets'


def save_dataset(data: pd.DataFrame,
                 folder_to_save: str,
                 file_name: str) -> None:

    create_folder_if_not_exists(folder_path=folder_to_save)
    saving_path = f'{folder_to_save}/{file_name}.parquet'
    data.to_parquet(path=saving_path,
                    index=False)


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
    loss_function_classes = {'mse': MeanSquaredError,
                             'mae': MeanAbsoluteError,
                             'msle': MeanSquaredLogarithmicError}

    try:
        selected_loss_function = loss_function_classes[loss_type, ]()
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


def get_callbacks():
    pass


if __name__ == '__main__':
    global_configuration = read_yaml_parameters(folder_path=configuration_path)
    matlab_configuration = global_configuration['matlab_data']
    dataset_configuration = global_configuration['dataset']
    neural_network_configuration = global_configuration['model']
    callbacks_configuration = global_configuration['callbacks']

    """
    We can create a dataset from zero, getting the data from a matlab file and converting into a dataframe or we can
    use an existing dataframe that we've created before.
    In case we create a new dataset, we can either use the raw dataset or the same dataset but with the data shuffled
    using sliding windows.
    """
    if dataset_configuration['create_dataset']:
        raw_dataset, shuffled_dataset = generate_datasets(matlab_configuration=matlab_configuration,
                                                          dataset_configuration=dataset_configuration)

        if dataset_configuration['save_dataset']:
            save_dataset(data=shuffled_dataset,
                         folder_to_save=datasets_path,
                         file_name=dataset_configuration['name'])

    furuta_pendulum_model = create_model_architecture(input_shape=neural_network_configuration['architecture']['input_shape'],
                                                      number_units=neural_network_configuration['architecture']['number_units'],
                                                      activation_hidden_layers=neural_network_configuration['architecture']['activation_hidden_layers'],
                                                      activation_output_layer=neural_network_configuration['architecture']['activation_output_layer'])

    optimizer = get_optimizer(optimizer_type=neural_network_configuration['optimizer']['optimizer_type'],
                              learning_rate=neural_network_configuration['optimizer']['learning_rate'])
    loss = get_loss_function(loss_type=neural_network_configuration['loss'])
    metrics = get_metrics(metrics_type=neural_network_configuration['metrics'])

    furuta_pendulum_model.compile(loss=loss,
                                  optimizer=optimizer,
                                  metrics=metrics)



