"""

This file is used to define the custom transform

Author: Kyrillos Botros
Date: Jul 26, 2023

"""

from monai.transforms import Compose, Resize, ToTensor, EnsureChannelFirst

# if you will use this file in a different directory,
# you need to change the import statement to be relative to the app directory
from src.preprocessor.preprocessing_utils import (LoadDcm,
                                                  CorrectingWindow,
                                                  DenoiseImage,
                                                  CropScaleImage
                                                  )


class CustomImageTransform:
    """
    This class is used to define the custom transform
    """

    def __init__(self):
        """
        This function is used to initialize the custom transform

        """
        self.transform = Compose([
            LoadDcm(),
            CorrectingWindow(),
            DenoiseImage(),
            CropScaleImage(is_scale=True, scale_factor=1 / 3071),
            ToTensor(),
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize((224, 224))
        ])

    def __call__(self, file_path):
        """
        This function is used to call the custom transform

            Args:
                file_path(Path): The DICOM file absolute path
        """
        return self.transform(file_path)
