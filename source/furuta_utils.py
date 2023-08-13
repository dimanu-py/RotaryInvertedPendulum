import yaml
from functools import wraps


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


def load_configuration_data(func):
    @wraps(func)
    def wrapper(self, folder_path, data_key):
        configuration_data = read_yaml_parameters(folder_path)
        try:
            selected_data = configuration_data[data_key]
            return func(self, selected_data)
        except KeyError as error:
            print(f'Impossible to find the key -> {error.args[0]}')
    return wrapper
