"""
This file is the main file for the FastAPI app.
It contains the API endpoints and the logic for the endpoints.

Author: Kyrillos Botros
Date: Jul 26, 2023

"""

import json
from io import BytesIO
import torch
from monai.transforms import Activations
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from src.model.model import HemorrhageModel
from src.preprocessor.custom_transform import CustomImageTransform


# Create the app
app = FastAPI()

# Load the model
model = HemorrhageModel.load_from_checkpoint("model/model-epoch=03-val_f1=0.82.ckpt", map_location=torch.device('cpu'))
model.eval()

# Create an image transform object
image_transform = CustomImageTransform()

# Create a list of classes
classes = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


def perform_inference(dicom_file):
    """
    This function performs inference on a single DICOM file.
    :param dicom_file: The DICOM file to perform inference on.
    """
    # Apply the transform to the image
    image = image_transform(dicom_file)
    with torch.no_grad():

        # predict labels
        pred = model(image.unsqueeze(0))[0]  #adding batch dimension
        pred = Activations(sigmoid=True)(pred) #applying sigmoid to the output
        pred = pred.cpu().numpy() >= 0.5 #converting to numpy array and thresholding
        preds = pred.astype(np.int8) #converting Trus and false returns intto int8

        # check if all preds are 0 or if "any" category equal 1 and all others are 0.
        # As if all subtypes are 0 with "any" category is 1,
        # then the model cannot detect any of subtypes and it wrongly predict "any" category.
        if np.sum(preds) == 0 or (preds[0]==1 and np.sum(preds)==1):
            return {"prediction": "normal"}

        abnormal_types = [classes[i] for i in range(1, preds.shape[0]) if preds[i]==1]
        return {"prediction": "anomalous", "abnormal_types": abnormal_types}

@app.get("/")
async def welcom():
    """
    This function is the endpoint for the API.
    It shows welcome message.

    Returns:
        JSONResponse: The response from the API.
    """
    welcome_message = {"message": "Welcome to the Hemorrhage Detection API!"}
    return JSONResponse(content=json.dumps(welcome_message), status_code=200)

@app.post("/predict")
async def predict(dicom_file: UploadFile = File(...)):
    """
    This function is the endpoint for the API.
    It performs inference on a single DICOM file.
    Parameters:
        dicom_file(File opject): The DICOM file to perform inference on.
    Returns:
        JSONResponse: The response from the API.
    """
    try:
        contents_binary = await dicom_file.read()
        inference_result = perform_inference(BytesIO(contents_binary))

        inference_result['image-path'] = dicom_file.filename
        return JSONResponse(content=json.dumps(inference_result), status_code=200)

    except Exception as error:
        error_message = {"error": str(error)}
        return JSONResponse(content=json.dumps(error_message), status_code=500)
