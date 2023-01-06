"""
**********************************************************************************************************
This program was developed by Kittiwat Pheramunchai and Shi-Jinn Horng in August 27, 2021, 
in the Information Security and Parallel Processing Laboratory which is located at 
National Taiwan University of Science and Technology. Do not copy of distribute it without permission.
**********************************************************************************************************
"""
import numpy as np
import cv2
from ela import ela
from morph import *
from PIL import Image

def hsvTest(input_image, window_name = "", rgb = False):

        if rgb == True:
            img_hsv = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
        else:
            img_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        if window_name:
            name = window_name
        else:
            name = "image"

        cv2.namedWindow("Trackbars", flags=cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"binary_{name}", flags=cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"bitwise_{name}", flags=cv2.WINDOW_NORMAL)
        cv2.createTrackbar("upper_H", "Trackbars", 180, 180, lambda tmp:None)
        cv2.createTrackbar("upper_S", "Trackbars", 255, 255, lambda tmp:None)
        cv2.createTrackbar("upper_V", "Trackbars", 255, 255, lambda tmp:None)
        cv2.createTrackbar("lower_H", "Trackbars", 0, 180, lambda tmp:None)
        cv2.createTrackbar("lower_S", "Trackbars", 0, 255, lambda tmp:None)
        cv2.createTrackbar("lower_V", "Trackbars", 0, 255, lambda tmp:None)
        print("Press 'w' to end the while loop")

        while (True):
            upper_H = cv2.getTrackbarPos("upper_H", "Trackbars")
            upper_S = cv2.getTrackbarPos("upper_S", "Trackbars")
            upper_V = cv2.getTrackbarPos("upper_V", "Trackbars")
            lower_H = cv2.getTrackbarPos("lower_H", "Trackbars")
            lower_S = cv2.getTrackbarPos("lower_S", "Trackbars")
            lower_V = cv2.getTrackbarPos("lower_V", "Trackbars")

            lower = np.array([lower_H, lower_S, lower_V], dtype = np.uint8)
            upper = np.array([upper_H, upper_S, upper_V], dtype = np.uint8) 
            binary_image = cv2.inRange(img_hsv,lower ,upper)
            bitwise_image = cv2.bitwise_and(input_image, input_image, mask = binary_image)
            cv2.imshow(f"binary_{name}", binary_image)
            cv2.imshow(f"bitwise_{name}", bitwise_image)

            if cv2.waitKey(1) & 0xFF == ord("w"):
                return binary_image
                # print("lower, upper = [{},{},{}], [{},{},{}]\n".format(lower[0],
                #                                                     lower[1],
                #                                                     lower[2],
                #                                                     upper[0],
                #                                                     upper[1],
                #                                                     upper[2]))
                break
        cv2.destroyWindow(f"binary_{name}")
        cv2.destroyWindow(f"bitwise_{name}")
        cv2.destroyWindow("Trackbars")

def postprocess(ela_image, rgb=False):
    if rgb == True:
        img_hsv = cv2.cvtColor(ela_image, cv2.COLOR_RGB2HSV)
    else:
        img_hsv = cv2.cvtColor(ela_image, cv2.COLOR_BGR2HSV)
    
    binary = cv2.inRange(img_hsv, np.array(
        [0, 95, 77], dtype=np.uint8), np.array([37, 255, 255], dtype=np.uint8))
    
    # cv2.namedWindow("morph_img", cv2.WINDOW_NORMAL)
    
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * 10 + 1, 2 * 10 + 1),(9, 9))
    morph_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    # cv2.imshow("morph_img", morph_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * 2 + 1, 2 * 2 + 1),(2, 2))
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, element)
    # cv2.namedWindow("morph_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("morph_img", morph_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * 2 + 1, 2 * 2 + 1),(2, 2))
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, element)
    # cv2.namedWindow("morph_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("morph_img", morph_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * 3 + 1, 2 * 3 + 1), (3, 3))
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, element)
    # cv2.namedWindow("morph_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("morph_img", morph_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * 1 + 1, 2 * 1 + 1), (1, 1))
    morph_img = cv2.erode(morph_img, element)
    # cv2.namedWindow("morph_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("morph_img", morph_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * 3 + 1, 2 * 3 + 1), (3, 3))
    morph_img = cv2.dilate(morph_img, element)
    # cv2.namedWindow("morph_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("morph_img", morph_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    gaussian = cv.GaussianBlur(morph_img, (41, 41), 0)
    # print(gaussian)
    # cv2.namedWindow("gaussian", cv2.WINDOW_NORMAL)
    # cv2.imshow("gaussian", gaussian)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    norm_image = cv2.normalize(gaussian, None, alpha=0.01, beta=1,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # print("==============================")
    # print(norm_image)
    norm_image_pil = Image.fromarray(norm_image).convert('RGB')
    
    return norm_image_pil
    

if __name__ == '__main__':
    # path = r"D:\Work\NTUST\Thesis\Codes\Ela-InceptionResnet\dataset\crop\ela\fake\video_0whiten_frame_50_ela.jpg"
    # img = cv2.imread(path)
    img = Image.open(r"C:\Thesis\FaceForensics-master\FaceForensics-master\classification\Kittiwat_dataset\dataset\crop\demo\fake\video_3whiten_frame_150.jpg")
    image = ela(img)
    postprocess(image)
    # postprocess(img)
    
    
    # binary = hsvTest(open_cv_image)
    # morph_img = closing(binary)
    # morph_img = closing(morph_img)
    # morph_img = opening(morph_img)
    # morph_img = closing(morph_img)
    # morph_img = erode(morph_img)
    # morph_img = dilate(morph_img)
    # blur_img = gaussian(morph_img)
    # norm_image = cv2.normalize(blur_img, None, alpha=0.01, beta=1,
    #                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # cv2.imshow("norm", norm_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
