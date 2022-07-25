# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                                  (module) main.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                           https://keepo.io/rp11ai ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import cv2
import motion_detector as md
# ---------------------------------------------------------------------------------------------------------------------|


def EXECUTE_OD() -> None:
    """
    Optimized motion detection method. It does not return images and generates few internal variables. Low processing
    cost. Ideal for optimizing systems that need to run some script when necessary.
    """
    OD = md.OptimizedDetection(directory=0, contour_area=5000)
    while OD.cap.isOpened():
        auth_list = OD.Frames()
        for i in auth_list:
            if not i:
                continue
        OD.Treatment()
        print(OD.__bool__())
        if cv2.waitKey(5) & 0xFF == 27:
            break
    OD.DestroyCap()


def EXECUTE_MD() -> None:
    """
    Full detection method. It has educational purposes about the opencv handling methods.
    """
    MD = md.MotionDetection(0, 1000, record_type='stack')
    while MD.cap.isOpened():
        auth_list = MD.Frames()
        for i in auth_list:
            if not i:
                continue
        MD.Treatment()
        MD.BoundingBoxDetection(print_bbox=True, sound=False)
        MD.GenerateStacking()
        MD.RecordVideo()
        if cv2.waitKey(5) & 0xFF == 27:
            break
    MD.DestroyCap()


if __name__ == '__main__':
    EXECUTE_MD()
