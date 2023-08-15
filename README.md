# Python-Furuta-Pendulum

To install quanser libraries run
```bash
 python -m pip install --upgrade --find-links "C:\Program Files\Quanser\QUARC\python" "C:\Program Files\Quanser\QUARC\python\quanser_api-2021.12.3-py2.py3-none-any.whl"
```

## Configure DVC project

1. Inside your project folder run
    ```bash
    dvc init
    ```
2. Add remote storage
    ```bash
    dvc remote add -d storage gdrive://id_of_your_gdrive_folder
    ```
3. Install dvc library for google drive
    ```bash
    pip install dvc-gdrive
    ```
4. Get the data from the remote storage
    ```bash
    dvc pull
    ```

