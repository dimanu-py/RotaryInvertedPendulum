import os

import yaml


def read_yaml_parameters(yaml_path: str = None) -> dict:
    """
    Read yaml file and load content in a dictionary.
    :param yaml_path: path to yaml file
    :return: content as a dictionary
    """
    if yaml_path is None:  # default path
        yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/parameters.yaml'))

    with open(yaml_path, 'r') as params_file:
        try:
            parameters = yaml.load(stream=params_file,
                                   Loader=yaml.SafeLoader)

            return parameters

        except Exception as e:
            print(f'Error loading yaml {e.args[1]}')


def extract_extension(file_name: str) -> str:
    """
    Extract file extension from a file name.
    :param file_name: name of the file
    :return: file extension
    """
    extension = file_name.split('.')[-1]
    return extension
