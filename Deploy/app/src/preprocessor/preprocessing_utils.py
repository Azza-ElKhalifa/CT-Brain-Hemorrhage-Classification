"""
This module contains the preprocessing functions used in the project

Author: Kyrillos Botros
Date: Jul 26, 2023

"""
from monai.transforms.compose import Transform
import pydicom
import numpy as np
import cv2


class LoadDcm(Transform):
    """

    This is class created to be used in monai.transforms.compose to load the DICOM object

    """

    def __init__(self):
        pass

    def __call__(self, dcm_file_path):
        """

        This function returns DICOM object that contains metadata and image

        Args:
            dcm_file_path(Path): The DICOM file absolute path

        Returns:
            DICOM object

        """
        return pydicom.dcmread(dcm_file_path)


class CorrectingWindow(Transform):
    """
    This class contains the helper functions needed to correct image windows

    """

    def __init__(self, window_level=None, window_width=None):
        """

         Args:
                window_level(int): Default is None.
                If None, the window level from metadata will be used

                window_width(int): Default is None.
                If None, the window width from metadata will be used

        """
        self.window_level = window_level
        self.window_width = window_width

    def get_first_dicom_field_int(self, value):
        """

        This function is used to get the first value of multiple
            values of DICOM metadata fields and convert it into integer

            Parameters:
                value ([float, pydicom multival.MultiValue, pydicom.valuerep.DSfloat]):
                Metadata field value

            Returns:
                value(int): Converted value into integer

        """
        if isinstance(value, pydicom.multival.MultiValue):
            return int(value[0])

        return int(value)

    def transform_hu(self, image, intercept, slope):
        """

        This function is used to transfer the image pixel values to Hounisfield Unit (HU)

            Parameters:
                image(np.array): It's a DICOM image

            Returns:
                HU_image(np.array): A new image with HU transformation

        """
        hu_image = image * slope + intercept

        return hu_image

    def window_image(self, image, window_level, window_width):
        """

        This function is used to adjust an image containing HU values
         with a certain range of gray shades density

            Parameters:
                image (np.array): An image with HU values
                window_level(int): The centre of the grayscale range
                window_width(int): The range of grayscale

            Returns:
                window_image(np.array): A new image adjusted with window leval and window width

        """
        lowest_value = window_level - window_width // 2
        highest_value = window_level + window_width // 2

        window_image = image.copy()

        window_image[window_image < lowest_value] = lowest_value
        window_image[window_image > highest_value] = highest_value

        return window_image

    def window_hu_image(self, dcm_file, window_level=None, window_width=None):
        """

        This function is used to view the adjusted image by transform_HU and window_image functions

            Parameters:

                dcm_file(DICOM): It's a DICOM file

                window_level(int): Windowlevel to be allied on the image
                                   if None, it will be the window_level of the DICOM file

                window_width(int): Windowwidth to be applied on the image
                                    if None, it will be the window_width of the DICOM file


            Returns:
                adjusted_image(np.array): An adjusted image after applying HU,
                window level and window width

        """
        if window_level and window_width:
            window_level_ = window_level
            window_width_ = window_width
        else:
            window_level_ = dcm_file.WindowCenter
            window_width_ = dcm_file.WindowWidth

        fields_list = [window_level_,
                       window_width_,
                       dcm_file.RescaleIntercept,
                       dcm_file.RescaleSlope
                       ]

        window_level, window_width, intercept, slope = \
            [self.get_first_dicom_field_int(value) for value in fields_list]

        image = dcm_file.pixel_array
        hu_image = self.transform_hu(image, intercept, slope)

        adjusted_image = self.window_image(
            hu_image, window_level, window_width)
        return adjusted_image

    def __call__(self, dcm_obj):
        """
        Executing window_hu_image function with required arguments

            Args:
                dcm_obj: DICOM Object

            Returns:
                adjusted_image (np.array)

        """
        return self.window_hu_image(
            dcm_obj, self.window_level, self.window_width)


class DenoiseImage(Transform):
    """
    This class contains functions to remove noise from image
    """

    def __init__(self, kernel_size=(3, 3)):
        """
             Args:
                 kernel_size(tuple): This is the kernel size used for dialation

        """
        self.kernel_size = kernel_size

    def denoise_image(self, image=None, kernel_size=None):
        """

        This function is used to remove the noise from the image

            Parameters:
                image(np.array): This image used to remove the noise from
                kernel_size(tuple): This is the kernel size used for dialation

            Returns:
                masked_image(np.array): This is the masked image

        """
        kernel = np.ones(kernel_size)

        segmentation = cv2.dilate(image.astype("uint8"), kernel)
        _, labels = cv2.connectedComponents(segmentation)

        # Getting the number of clsases or segmentation found
        label_count = np.bincount(labels.ravel().astype("int8"))

        # the first class is background, we won't use it
        label_count[0] = 0

        mask = labels == label_count.argmax()
        mask = cv2.dilate(mask.astype("uint8"), kernel)
        masked_image = mask * image

        return masked_image

    def __call__(self, image):
        """

        This function to execute denoise_image function

            Parameters:
                image(np.array): This image used to remove the noise from

            Returns:
                masked_image(np.array): This is the masked image

        """
        return self.denoise_image(image=image, kernel_size=self.kernel_size)


class CropScaleImage(Transform):
    """

    This class contains functions to crop image

    """

    def __init__(self, is_scale=False, scale_factor=1):
        """
         Args:
            is_scale(bool): to scale the image
            scale_factor(float): scaling image with this number if is_scale = True

        """
        self.is_scale = is_scale
        self.scale_factor = scale_factor

    def crop_scale_image(self, image, is_scale=False, scale_factor=1):
        """

        This functio is used to crop the image to filt the object in the center

            Parameters:
                image(np.array): This is the image needed to be cropped
                is_scale(bool): to scale the image
                scale_factor(float): scaling image with this number if is_scale = True

            Returns:
                cropped_image(np.array): This is the final output

        """
        # Create a mask with the background pixels
        mask = image == 0

        # gitting the coordinates of the object which pixel values !=0
        coords = np.transpose(np.nonzero(~mask))
        # Find the object area
        if coords.size != 0:
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)

            # Crop image
            cropped_image = image[x_min:x_max,
                                  y_min:y_max]
        else:
            cropped_image = image
        if is_scale:
            cropped_image = cropped_image.astype("float32") * scale_factor

        return cropped_image

    def __call__(self, image):
        """

        This function to execute crop_image function

            Parameters:
                image(np.array): This is the image needed to be cropped

            Returns:
                cropped_image(np.array): This is the final output

        """
        return self.crop_scale_image(
            image=image,
            is_scale=self.is_scale,
            scale_factor=self.scale_factor)
