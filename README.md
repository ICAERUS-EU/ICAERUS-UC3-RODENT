# ICAERUS-RODENT
Repo for Rodent Obstruction through Drone-Enabled Non-invasive Technology project.

## Structure
This repository contains:
- testing_SPL/
    - **rezultati_merenja.m** loads measured signals from signali/ with filenames of the form elevacija_x.mat where x ranges from 1 to 30. These signals were recorded on the surface beneath the parabola to assess sound-field coverage. The script computes the sound pressure level (SPL) on the surface located 1.0 m below the parabola's center, for a parabola of radius 0.58 m, and summarises the spatial SPL distribution used for coverage analysis. The signali/ dataset is hosted on the Zenodo platform (DOI/URL in progress).
- infrared-rodent-detection/
    - **finetune_yolo.ipynb** contains the finetuning script for the YOLOv11n model. The model was trained on this [dataset](https://universe.roboflow.com/panav2/rodent-thermal/dataset/2), which is also uploaded in the according format on the Zenodo platform (DOI/URL in progress).
    - **gpu-yolo.yml** is the environment YAML file which was used for finetuning the YOLOv11n model on GPU.
    - **realtime_yolo.py** contains the script for running YOLO realtime inference on images streamed from an IR thermal camera.
    - **spin-yolo.yml** is the environment YAML file which is required for running YOLO inference and connecting to the IR camera. The environment requires the Spinnaker SDK to be installed on the system, as well as the PySpin package in Python.
    - The trained model and dataset (available on the Zenodo platform) should be placed under this folder.
