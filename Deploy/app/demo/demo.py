"""

This file is a demo file for the FastAPI app. It contains the code
to perform inference on a single DICOM file.

Author: Kyrillos Botros
Date: Jul 26, 2023

"""

import requests
# port = 5000
URL = "http://localhost:5000/predict" ## this URL to be changed with the server URL
DICOM_PATH = "examples/ID_fffc60817.dcm" ## this path to be changed with the path of the DICOM file
HEADER = "application/dicom"

with open(DICOM_PATH, "rb") as file:
    files = {"dicom_file": (DICOM_PATH, file, HEADER)}
    response = requests.post(URL, files=files, timeout=15)

    if response.status_code == 200:
        result_json = response.json()
        print("Inference Result:", result_json)

    else:
        print("Error occurred during inference.")
        print(response.json())
