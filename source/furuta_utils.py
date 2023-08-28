import yaml
import os
import matplotlib.pyplot as plt


def read_yaml_parameters(folder_path: str, yaml_file: str = None) -> dict:
    """
    Read yaml file and load content in a dictionary.
    """
    if yaml_file is None:
        yaml_path = f'{folder_path}/parameters.yaml'
    else:
        yaml_path = f'{folder_path}/{yaml_file}'

    try:
        with open(yaml_path, 'r') as params_file:
            parameters = yaml.load(stream=params_file,
                                   Loader=yaml.SafeLoader)

            return parameters
    except FileNotFoundError as error:
        print(f'Impossible to find the yaml file -> {error.args[1]}')


def extract_extension(file_name: str) -> str:
    """
    Extract file extension from a file name.
    """
    extension = file_name.split('.')[-1]
    return extension


def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Check if the folder exists. If not, create it.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def plot_training(history,
                  metric):
    data = history.history.get(metric)
    validation_data = history.history.get(f'val_{metric}')
    epochs = range(1, len(data) + 1, 1)

    plt.plot(epochs, data, 'r--', label=f'Training {metric}')
    plt.plot(epochs, validation_data, 'b', label=f'Validation {metric}')
    plt.ylabel(f'{metric}')
    plt.xlabel('epochs')
    plt.title(f'Training {metric} vs Validation {metric}')

    plt.legend()
    plt.show()
