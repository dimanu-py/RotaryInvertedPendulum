import pytest

from source.furuta_utils import read_yaml_parameters


def test_type_error_read_yaml_parameters():
    with pytest.raises(TypeError):
        read_yaml_parameters(yaml_path=None)


def test_read_yaml_parameters():
    yaml_path = r'C:\PROGRAMACION\PENDULO INVERTIDO\Pendulo Invertido Diego\Python-Furuta-Pendulum\test\yaml_test.yaml'
    parameters = read_yaml_parameters(yaml_path=yaml_path)
    assert isinstance(parameters, dict)
