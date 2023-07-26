from source.env_settings import EnvSettings


def test_get_env_vars():
    """
    Tests that .env file variables are retrieved correctly
    """
    env_vars = EnvSettings.get_env_vars()

    assert env_vars is not None
