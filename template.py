import os
from pathlib import Path

list_of_files=[
".github/workflows/.gitkeep",
"src/__init__.py",
"src/components/__init__.py",
"src/components/data_ingestion.py",
"src/components/data_transformation.py",
"src/components/model_training.py",
"src/components/model_monitoring.py",
"src/pipelines/__init__.py",
"src/pipelines/training.py",
"src/pipelines/prediction.py",
"requirements.txt",
"setup.py",
"src/logger.py",
"src/exception.py",
"main.py",
"app.py",
"src/utils.py",
"experiment/experiment.ipynb",
"setup.cfg",
"init_setup.sh",
"tests/unit/__init__.py",
"tests/integration/__init__.py",
"tox.ini",
"pyproject.toml",
"requirements_dev.txt"

]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    if filedir !="":
        os.makedirs(filedir,exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass