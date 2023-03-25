import yaml


def read_yaml_parameters(yaml_path=None) -> dict:

    if yaml_path is None:
        raise TypeError('yaml path must not be None')

    with open(yaml_path, 'r') as params_file:
        try:
            parameters = yaml.load(stream=params_file,
                                   Loader=yaml.SafeLoader)

            return parameters

        except Exception as e:
            print(e)
