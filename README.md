
# Intracranial Hemorrhage (ICH) Detection Project

This project detects subtypes of intracranial hemorrhages (ICH) from medical images using deep learning techniques. The project aims to aid in identifying and localizing various types of hemorrhages, including epidural, intraparenchymal, intraventricular, subarachnoid, and subdural hemorrhages, to support timely medical interventions.

## Table of Contents
1. [Installation](#installation)
2. [Dependencies](#dependencies)
3. [Task Description](#task-description)
4. [Usage](#usage)

## Installation

To set up the environment, ensure you have Python installed, and then install the necessary packages:

```bash
# Install required packages
pip install tqdm numpy pandas pydicom pytorch-lightning monai torchmetrics scikit-learn matplotlib seaborn opencv-contrib-python lightning tensorboard tensorboardx
```

Alternatively, you can use a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

> **Note**: To download wights used in the project, download it from [here](https://drive.google.com/drive/folders/1GkidPz9EwtPNtG4zLzQCSP7TnHWQMZ2d?usp=sharing).

## Dependencies

This project requires the following libraries:

- **Pandas**: For handling and manipulating data
- **NumPy**: For numerical operations and array manipulation
- **PyTorch Lightning**: For structured deep learning models
- **MONAI**: Medical imaging transformations and utilities
- **TorchMetrics**: For model performance metrics
- **Pydicom**: For handling DICOM files, a standard format for medical images
- **Scikit-learn**: For data preprocessing and model evaluation
- **Matplotlib and Seaborn**: For data visualization
- **OpenCV**: For advanced image processing, particularly for bounding boxes and data augmentation

Ensure all dependencies are installed before running the notebook.

## Task Description

The goal of this notebook is to develop a robust model for detecting and classifying intracranial hemorrhages from DICOM images. The project includes:

- **Data Preprocessing**: Includes custom classes for DICOM loading, windowing, denoising, and cropping.
- **Data Augmentation**: Transformations including random zoom, affine transformations, and resizing.
- **Model Training**: Multiple architectures are tested, including DenseNet and EfficientNet models.
- **Evaluation**: The model is evaluated using F1-score, specifically tuned to handle class imbalance in hemorrhage subtypes.

This project tackles the challenge of class imbalance by balancing the training data and applying weighted losses. The primary goal is to maximize the model’s F1-score, which reached 82% in the best configuration.

## Usage

1. Clone this repository and navigate to the directory.
2. Run the notebook in a Jupyter environment:

   ```bash
   jupyter notebook hemorrhagedetction.ipynb
   ```

3. Follow the cells to preprocess the data, train the model, and evaluate its performance.

> **Note**: This model is a tool to support medical professionals in detecting hemorrhages in imaging data and should not replace clinical judgment.

## License

This project is open-source and available under the MIT License.
