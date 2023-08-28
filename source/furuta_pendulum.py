from typing import List, Dict, Any

from tensorflow import keras
from keras_tuner.engine.hyperparameters import HyperParameters
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split

from source.dataset_generator import create_shuffled_dataset
from source.furuta_utils import (read_yaml_parameters,
                                 create_folder_if_not_exists,
                                 plot_training)

configuration_path = '../config'
datasets_path = '../data/datasets'
models_path = '../data/models'
results_path = '../data/results'


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


def create_model(input_shape,
                 number_units,
                 activation_hidden_layers,
                 activation_output_layer,
                 loss_type,
                 metrics_type,
                 optimizer_type,
                 learning_rate,
                 regularizer_type,
                 regularizer_value,
                 dropout) -> "Sequential":

    model = Sequential()

    model.add(Input(shape=(input_shape, )))

    regularizer = get_regularizer(regularizer_type=regularizer_type,
                                  regularizer_value=regularizer_value)

    for layer_index, layer_units in enumerate(number_units):
        model.add(Dense(units=layer_units,
                        kernel_regularizer=regularizer,
                        activation=activation_hidden_layers))
        model.add(Dropout(dropout))
    model.add(Dense(units=1,
                    activation=activation_output_layer))

    loss = get_loss_function(loss_type=loss_type)
    metrics = get_metrics(metrics_type=metrics_type)
    optimizer = get_optimizer(optimizer_type=optimizer_type,
                              learning_rate=learning_rate)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    return model


def get_regularizer(regularizer_type: str,
                    regularizer_value: float) -> "keras.regularizers":
    regularizer_classes = {'l1': keras.regularizers.l1,
                           'l2': keras.regularizers.l2,
                           'l1_l2': keras.regularizers.l1_l2}

    try:
        selected_regularizer = regularizer_classes[regularizer_type](regularizer_value)
        return selected_regularizer
    except KeyError:
        print(f'Regularizer {regularizer_type} not implemented yet.')


def get_optimizer(optimizer_type: str,
                  learning_rate: float) -> "keras.optimizers":

    optimizer_classes = {'adam': keras.optimizers.Adam,
                         'sgd': keras.optimizers.SGD,
                         'rmsprop': keras.optimizers.RMSprop}

    try:
        selected_optimizer = optimizer_classes[optimizer_type](learning_rate=learning_rate)
        return selected_optimizer
    except KeyError:
        print(f'Optimizer {optimizer_type} not implemented yet.')


def get_loss_function(loss_type: List[str]) -> "keras.losses":
    loss_type, = loss_type
    loss_function_classes = {'mse': keras.losses.MeanSquaredError,
                             'mae': keras.losses.MeanAbsoluteError,
                             'msle': keras.losses.MeanSquaredLogarithmicError}

    try:
        selected_loss_function = loss_function_classes[loss_type]()
        return selected_loss_function
    except KeyError:
        print(f'Loss function {loss_type} not implemented yet.')


def get_metrics(metrics_type: List[str]) -> List["keras.metrics"]:
    metrics_classes = {'mae': keras.metrics.MeanAbsoluteError,
                       'mse': keras.metrics.MeanSquaredError,
                       'rmse': keras.metrics.RootMeanSquaredError,
                       'mape': keras.metrics.MeanAbsolutePercentageError,
                       'msle': keras.metrics.MeanSquaredLogarithmicError,
                       'r2': keras.metrics.R2Score}

    try:
        selected_metrics = [metrics_classes[metric]() for metric in metrics_type]
        return selected_metrics
    except KeyError:
        print(f'Metrics {metrics_type} not implemented yet.')


def get_callbacks(callback_type: str, configuration: Dict[str, Any]):
    callbacks_classes = {'early_stopping': keras.callbacks.EarlyStopping,
                         'model_checkpoint': keras.callbacks.ModelCheckpoint,
                         'reduce_lr': keras.callbacks.ReduceLROnPlateau,
                         'tensor_board': keras.callbacks.TensorBoard}
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
    try:
        training_dataset = pd.read_parquet(path=f"{datasets_path}/{dataset_configuration_training['name']}.parquet")
    except (FileNotFoundError, FileExistsError):
        training_dataset = create_shuffled_dataset(matlab_configuration=matlab_configuration_training,
                                                   dataset_configuration=dataset_configuration_training)

    if dataset_configuration_training['save_dataset']:
        save_dataset(data=training_dataset,
                     folder_to_save=datasets_path,
                     file_name=dataset_configuration_training['name'])

    # CREATE OR LOAD DEV AND TEST DATASET
    try:
        dev_test_dataset = pd.read_parquet(path=f"{datasets_path}/{dataset_configuration_dev_test['name']}.parquet")
    except (FileNotFoundError, FileExistsError):
        dev_test_dataset = create_shuffled_dataset(matlab_configuration=matlab_configuration_dev_test,
                                                   dataset_configuration=dataset_configuration_dev_test)

    if dataset_configuration_dev_test['save_dataset']:
        save_dataset(data=dev_test_dataset,
                     folder_to_save=datasets_path,
                     file_name=dataset_configuration_dev_test['name'])

    # CREATE THE MODEL
    furuta_pendulum_model = create_model(input_shape=neural_network_configuration['architecture']['input_shape'],
                                         number_units=neural_network_configuration['architecture']['number_units'],
                                         activation_hidden_layers=neural_network_configuration['architecture']['activation_hidden_layers'],
                                         activation_output_layer=neural_network_configuration['architecture']['activation_output_layer'],
                                         loss_type=neural_network_configuration['loss'],
                                         metrics_type=neural_network_configuration['metrics'],
                                         optimizer_type=neural_network_configuration['optimizer']['optimizer_type'],
                                         learning_rate=neural_network_configuration['optimizer']['learning_rate'],
                                         regularizer_type=neural_network_configuration['architecture']['regularization']['type'],
                                         regularizer_value=neural_network_configuration['architecture']['regularization']['penalty'],
                                         dropout=neural_network_configuration['architecture']['dropout'])

    # SET CALLBACKS
    callbacks = [get_callbacks(callback_type, configuration) for callback_type, configuration in callbacks_configuration.items()]

    # LOAD MODEL IF WE WANT TO RE-TRAIN
    if training_configuration['retrain']:
        furuta_pendulum_model.load_weights(callbacks_configuration['model_checkpoint']['filepath'])

    # SEPARATE FEATURES AND TARGETS FROM TRAINING, DEV AND TEST SETS
    train_features, train_target = training_dataset[training_configuration['features']], training_dataset.pop(*training_configuration['target'])
    dev_test_features, dev_test_targets = dev_test_dataset[training_configuration['features']], dev_test_dataset.pop(*training_configuration['target'])

    dev_features, test_features, dev_target, test_target = train_test_split(dev_test_features,
                                                                            dev_test_targets,
                                                                            test_size=training_configuration['test_size'],
                                                                            random_state=73,
                                                                            shuffle=False)
    # TRAIN THE NEURAL NETWORK
    history = furuta_pendulum_model.fit(train_features.values,
                                        train_target.values,
                                        epochs=training_configuration['epochs'],
                                        batch_size=training_configuration['batch_size'],
                                        validation_data=(dev_features, dev_target),
                                        callbacks=callbacks,
                                        verbose=1,
                                        shuffle=False)

    scores = furuta_pendulum_model.evaluate(test_features,
                                            test_target)
    scores_df = pd.DataFrame({'loss': scores[0],
                              'mae': scores[1]}, index=['results'])

    predictions = furuta_pendulum_model.predict(test_features)

    comparison = pd.DataFrame({'predictions': predictions.flatten(),
                               'targets': test_target})
    comparison = comparison.reset_index(drop=True)

    # SAVE MODEL IN DATA FOLDER
    save_model(model=furuta_pendulum_model,
               folder_to_save=models_path,
               name=training_configuration['model_name'])

    comparison.to_csv(f'{results_path}/{training_configuration["model_name"]}_comparison.csv')
    scores_df.to_csv(f'{results_path}/{training_configuration["model_name"]}_scores.csv', index=False)
    plot_training(history, 'loss')
    # model = SVR()
    #
    # # GRID SEARCH
    # kernel = ['linear', 'poly', 'rbf', 'sigmoid']  # type of kernel when projecting data into higher dimension
    # tolerance = [1e-3, 1e-4, 1e-5, 1e-6]  # tolerance for stopping criterion
    # C = [1, 1.5, 2, 3.5, 10]  # regularization parameter
    # grid = dict(kernel=kernel, tol=tolerance, C=C)
    #
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='neg_mean_squared_error')
    # grid_result = grid_search.fit(train_features, train_target)
    # best_model = grid_result.best_estimator_
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # print("R2 score: %f" % best_model.score(test_features, test_target))
    #
    # # RANDOM SEARCH
    # kernel = ['linear', 'poly', 'rbf', 'sigmoid']  # type of kernel when projecting data into higher dimension
    # tolerance = loguniform(1e-6, 1e-3)  # tolerance for stopping criterion
    # C = [1, 1.5, 2, 3.5, 10]  # regularization parameter
    # grid = dict(kernel=kernel, tol=tolerance, C=C)
    #
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # random_search = RandomizedSearchCV(estimator=model, param_distributions=grid, n_jobs=-1, cv=cv, scoring='neg_mean_squared_error')
    # random_result = random_search.fit(train_features, train_target)
    # best_model = random_result.best_estimator_
    # print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
    # print("R2 score: %f" % best_model.score(test_features, test_target))