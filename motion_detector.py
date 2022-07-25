# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                       (module) motion_detector.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                           https://keepo.io/rp11ai ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
from typing import Union, Tuple, List
import cv2
import winsound
import numpy as np
# ---------------------------------------------------------------------------------------------------------------------|


class OptimizedDetection(object):
    def __init__(self, directory: Union[str, int], contour_area: int) -> None:
        """
        Start the class and define directory (video/camera) and also the detection sensitivity.
        :param directory: 0, 1, 2, etc. to display video devices on the computer or directory of a video file.
        :param contour_area: Detection sensitivity. The higher the value, the more difficult it is to detect movement.
        """
        self.frame_1, self.frame_2, self.contours = None, None, None
        self.directory = directory
        self.contour_area = contour_area
        self.cap = cv2.VideoCapture(directory, cv2.CAP_DSHOW)

    def Frames(self) -> List:
        """
        Starts detection of frames from file or video device. It must be inside a loop key to run correctly.
        :return: Returns a list containing 2 boolean values referring to the frame capture.
        """
        auth_1, self.frame_1 = self.cap.read()
        auth_2, self.frame_2 = self.cap.read()

        return [auth_1, auth_2]

    def Treatment(self,
                  gaussian_blur_ksize: Tuple = (5, 5),
                  gaussian_blur_sigmaX: float = 0,
                  thresh: float = 20,
                  thresh_maxval: int = 255,
                  dilated_iterations: int = 3) -> None:
        """
        Treatment of frame_1 and frame_2 for motion detection. Carefully read each parameter specification to get a
        better result. USAGE INSIDE A LOOP
             cv2.absdiff: Calculates the per-element absolute difference between two arrays or between an array and a
                           scalar.
            cv2.cvtColor: Converts an image from one color space to another
        cv2.GaussianBlur: Blurs an image using a Gaussian filter.
           cv2.threshold: Applies a fixed-level threshold to each array element.
              cv2.dilate: Dilates an image by using a specific structuring element.
        cv2.findContours: Finds contours in a binary image.

        :param gaussian_blur_ksize: ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both
                                    must be positive and odd. Or, they can be zero's, and then they are computed from
                                    sigma.
        :param gaussian_blur_sigmaX: Gaussian kernel standard deviation in X direction.
        :param thresh: The function applies fixed-level thresholding to a multiple-channel array. The function is
                       typically used to get a bi-level (binary) image out of a grayscale image (#compare could be
                       also used for this purpose) or for removing a noise, that is, filtering out pixels with too
                       small or too large values. There are several types of thresholding supported by the function.
                       They are determined by type parameter.
        :param thresh_maxval: maximum value to use with the #THRESH_BINARY and #THRESH_BINARY_INV thresholding types.
        :param dilated_iterations: number of times dilation is applied.
        """
        blur = cv2.GaussianBlur(src=cv2.cvtColor(src=cv2.absdiff(src1=self.frame_1, src2=self.frame_2),
                                                 code=cv2.COLOR_RGB2GRAY),
                                ksize=gaussian_blur_ksize,
                                sigmaX=gaussian_blur_sigmaX)
        _, thresh = cv2.threshold(src=blur, thresh=thresh, maxval=thresh_maxval, type=cv2.THRESH_BINARY)
        self.contours, _ = cv2.findContours(image=cv2.dilate(src=thresh,
                                                             kernel=None,
                                                             iterations=dilated_iterations),
                                            mode=cv2.RETR_TREE,
                                            method=cv2.CHAIN_APPROX_NONE)

    def __bool__(self):
        """
        Suitable function to know when there is a motion detection. USAGE INSIDE A LOOP.
        :return: Boolean.
        """
        for c in self.contours:
            if cv2.contourArea(c) < self.contour_area:
                continue
            return bool(self.contours)

    def DestroyCap(self) -> None:
        """
        Releases the camera from using video and destroys all windows generated by opencv.
        """
        self.cap.release()
        cv2.destroyAllWindows()


class MotionDetection(object):
    def __init__(self, directory: Union[str, int], contour_area: int, record_type: str = 'stack') -> None:
        """
        Start the class and define directory (video/camera) and also the detection sensitivity.
        :param directory:  0, 1, 2, etc. to display video devices on the computer or directory of a video file.
        :param contour_area: Detection sensitivity. The higher the value, the more difficult it is to detect movement
        :param record_type: Whether to record the video. 'stack' to record the detection system and 'default' if only
                            the detection should be recorded.
        """
        self.frame_1, self.frame_2, self.contours = None, None, None
        self.directory = directory
        self.contour_area = contour_area
        self.diff, self.gray, self.blur, self.thresh, self.dilated = None, None, None, None, None
        self.cap = cv2.VideoCapture(directory, cv2.CAP_DSHOW)

        _, img = self.cap.read()
        if record_type == 'stack':
            self.resolution: Tuple = (int(img.shape[1] * 3), int(img.shape[0] * 2))
        elif record_type == 'default':
            self.resolution: Tuple = (img.shape[1], img.shape[0])

        self.record_type = record_type
        self.stack = None
        self.output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, self.resolution)

    @staticmethod
    def stack_images(scale: float, img_array: Union[Tuple, List]) -> np.ndarray:
        """
            Def to generate a generic image stack. Within the Detector object, it has the function of generation a grid
            [2 Lines, X Columns] where X is the 'box_per_line' parameter.
            :param scale: Percent resizing of image resolution.
            :param img_array: List or array containing images to be stacked.
            :return: Image grid.
            """
        rows, cols = len(img_array), len(img_array[0])
        width, height = img_array[0][0].shape[1], img_array[0][0].shape[0]
        if isinstance(img_array[0], list):
            for x in range(0, rows):
                for y in range(0, cols):
                    if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                        img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                    else:
                        img_array[x][y] = cv2.resize(img_array[x][y],
                                                     (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                     None, scale, scale)
                    if len(img_array[x][y].shape) == 2:
                        img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
            hor = [np.zeros((height, width, 3), np.uint8)] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(img_array[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if img_array[x].shape[:2] == img_array[0].shape[:2]:
                    img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
                else:
                    img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None,
                                              scale, scale)
                    if len(img_array[x].shape) == 2:
                        img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
            ver = np.hstack(img_array)
        return ver

    def Frames(self) -> List:
        """
        Starts detection of frames from file or video device. It must be inside a loop key to run correctly.
        :return: Returns a list containing 2 boolean values referring to the frame capture.
        """
        auth_1, self.frame_1 = self.cap.read()
        auth_2, self.frame_2 = self.cap.read()
        return [auth_1, auth_2]

    def Treatment(self,
                  gaussian_blur_ksize: Tuple = (5, 5),
                  gaussian_blur_sigmaX: float = 0,
                  thresh: float = 20,
                  thresh_maxval: int = 255,
                  dilated_iterations: int = 3) -> None:
        """
        Treatment of frame_1 and frame_2 for motion detection. Carefully read each parameter specification to get a
        better result. USAGE INSIDE A LOOP
             cv2.absdiff: Calculates the per-element absolute difference between two arrays or between an array and a
                          scalar.
            cv2.cvtColor: Converts an image from one color space to another
        cv2.GaussianBlur: Blurs an image using a Gaussian filter.
           cv2.threshold: Applies a fixed-level threshold to each array element.
              cv2.dilate: Dilates an image by using a specific structuring element.
        cv2.findContours: Finds contours in a binary image.

        :param gaussian_blur_ksize: ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both
                                    must be positive and odd. Or, they can be zero's, and then they are computed from
                                    sigma.
        :param gaussian_blur_sigmaX: Gaussian kernel standard deviation in X direction.
        :param thresh: The function applies fixed-level thresholding to a multiple-channel array. The function is
                       typically used to get a bi-level (binary) image out of a grayscale image (#compare could be
                       also used for this purpose) or for removing a noise, that is, filtering out pixels with too
                       small or too large values. There are several types of thresholding supported by the function.
                       They are determined by type parameter.
        :param thresh_maxval: maximum value to use with the #THRESH_BINARY and #THRESH_BINARY_INV thresholding types.
        :param dilated_iterations: number of times dilation is applied.
        """
        self.diff = cv2.absdiff(src1=self.frame_1, src2=self.frame_2)
        self.gray = cv2.cvtColor(src=self.diff, code=cv2.COLOR_RGB2GRAY)
        self.blur = cv2.GaussianBlur(src=self.gray, ksize=gaussian_blur_ksize, sigmaX=gaussian_blur_sigmaX)
        _, self.thresh = cv2.threshold(src=self.blur, thresh=thresh, maxval=thresh_maxval, type=cv2.THRESH_BINARY)
        self.dilated = cv2.dilate(src=self.thresh, kernel=None, iterations=dilated_iterations)
        self.contours, _ = cv2.findContours(image=self.dilated, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    def BoundingBoxDetection(self, print_bbox: bool = False, sound: bool = False) -> None:
        """
        Generates a bounding box where there is some movement and emits sound. USAGE INSIDE A LOOP.
        :param print_bbox: Boolean
        :param sound: Boolean
        """
        if print_bbox:
            for c in self.contours:
                if cv2.contourArea(c) < self.contour_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if sound:
                    winsound.Beep(frequency=5000, duration=5)
                cv2.rectangle(img=self.frame_1, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=1)

    def GenerateStacking(self) -> None:
        """
        It will generate an image stack containing each imaging process needed to identify a detection. USASE INSIDE A
        LOOP.
        """
        img_list = [self.diff, self.gray, self.blur, self.thresh, self.dilated]
        nam_list = ['ABS_DIFF', 'GRAY_CONVERT', 'BLUR', 'THRESH', 'DILATED']
        var_list = ['diff_text', 'gray_text', 'blur_text', 'thresh_text', 'dilated_text']
        img_dict = {}
        for n, img in enumerate(img_list):
            img_dict[var_list[n]] = cv2.putText(img, nam_list[n], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 255, 255), 1)
        self.stack = MotionDetection.stack_images(
            1, ([img_dict['diff_text'], img_dict['gray_text'], img_dict['blur_text']],
                [img_dict['thresh_text'], img_dict['dilated_text'], self.frame_1])
        )

    def __bool__(self):
        """
        Suitable function to know when there is a motion detection. USAGE INSIDE A LOOP.
        :return: Boolean.
        """
        for c in self.contours:
            if cv2.contourArea(c) < self.contour_area:
                continue
            return bool(self.contours)

    def ShowVideo(self, mode: int = 0) -> None:
        """
        Shows a window containing the processed images. USAGE INSIDE A LOOP.
        :param mode: 0 to show detection only; 1 to show the real-time imaging process; 2 to show both.
        """
        if mode == 0:
            cv2.imshow('Motion Detector', self.frame_1)
        if mode == 1:
            cv2.imshow('Stack_image', self.stack)
        if mode == 2:
            cv2.imshow('Motion Detector', self.frame_1)
            cv2.imshow('Stack_image', self.stack)

    def RecordVideo(self):
        """
        Function to record each frame. USAGE INSIDE A LOOP.
        :return:
        """
        if self.record_type == 'stack':
            self.stack = cv2.resize(self.stack, dsize=self.resolution)
            self.output.write(self.stack)
        elif self.record_type == 'default':
            self.frame_1 = cv2.resize(self.frame_1, dsize=self.resolution)
            self.output.write(self.frame_1)

    def DestroyCap(self):
        """
        Releases the camera from using video and destroys all windows generated by opencv.
        """
        self.cap.release()
        cv2.destroyAllWindows()
