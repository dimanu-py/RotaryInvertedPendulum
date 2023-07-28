import yaml


def read_yaml_parameters(folder_path: str) -> dict:
    """
    Read yaml file and load content in a dictionary.
    """
    yaml_path = f'{folder_path}/parameters.yaml'
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
