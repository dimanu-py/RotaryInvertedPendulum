import pytest

from source.furuta_utils import read_yaml_parameters


@pytest.mark.parametrize('yaml_path', [r'C:\PROGRAMACION\PENDULO INVERTIDO\Pendulo Invertido Diego\Python-Furuta-Pendulum\test\yaml_test.yaml'])
def test_read_yaml_parameters(yaml_path):
    parameters = read_yaml_parameters(yaml_path=yaml_path)

    assert isinstance(parameters, dict)

