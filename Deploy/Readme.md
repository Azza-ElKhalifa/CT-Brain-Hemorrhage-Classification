# Project
## Description

This project is a demo for detecting intracranial hemorrhage in CT DICOM images, 
by sending a post request to the app with a DICOM file, the app will return whether 
this patient has intracranial hemorrhage or not. in addition to the type of hemorrhage 
if it exists.

## Project Structure
```
├──Dockerfile
├── app
│   ├── demo
│   │   ├── demo.py
│   │   ├── examples 
│   │   │   ├── ID_000000e27.dcm
│   │   │   ├── ID_fffc60817.dcm
│   └── model
│   │   ├── model-epoch=03-val_f1=0.82.ckpt
│   ├── src
│   │   ├── model
│   │   │   ├── model.py
|   │   ├── preprocessor
│   │   │   ├── custom_transforms.py
│   │   │   ├── preprocessing_utils.py
│   ├── main.py
│   ├── requirements.txt

```
## Installation
### Local
- To run the app locally, open the terminal in the app folder and insert
```
pip install -r requirements.txt
```
- The app listens on port 5000, to run the app insert
```
uvicorn main:app --host 0.0.0.0 --port 5000
```
- Now you can use the app by sending a post request to the app with a DICOM file.
**See the demo.py that exists in demo directory to know the way of sending this file**

### Docker
**It may take up to 15 minutes to build the image**

- To run the docker file and see the logs of image built, 
open the terminal in the Deploy folder and insert
```
docker build -t <image_name> --progress=plain .
```
- The app listens on port 5000, to run the docker image insert
```
docker run -it -d --name <container_name> -p 5000:5000 <image_name>
```
- Now you can use the app by sending a post request to the app with a DICOM file.
**See the demo.py that exists in demo directory to know the way of sending this file**


## Dependencies
- fastapi==0.100.0
- python-multipart==0.0.6
- uvicorn==0.23.1
- monai==1.2.0
- numpy==1.24.1
- opencv_contrib_python==4.8.0.74
- pydicom==2.4.2
- pytorch_lightning==2.0.5
- Requests==2.31.0
- torch==2.0.1
- torchmetrics==1.0.1





