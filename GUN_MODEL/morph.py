"""
**********************************************************************************************************
This program was developed by Kittiwat Pheramunchai and Shi-Jinn Horng in August 27, 2021, 
in the Information Security and Parallel Processing Laboratory which is located at 
National Taiwan University of Science and Technology. Do not copy of distribute it without permission.
**********************************************************************************************************
"""
import cv2 as cv
import numpy as np


def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE


def closing(image):
    src = image
    cv.namedWindow("closing", cv.WINDOW_NORMAL)
    cv.createTrackbar("kernel shape",
                      "closing", 0, 2, lambda tmp: None)
    cv.createTrackbar("kernel size",
                      "closing", 0, 100, lambda tmp: None)
    while (True):
        kernel_size = cv.getTrackbarPos(
            "kernel size", "closing")
        kernel_shape = morph_shape(cv.getTrackbarPos(
            "kernel shape", "closing"))
        element = cv.getStructuringElement(kernel_shape, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                            (kernel_size, kernel_size))
        closing = cv.morphologyEx(src, cv.MORPH_CLOSE, element)
        cv.imshow("closing", closing)
        if cv.waitKey(1) & 0xFF == ord("w"):
            cv.destroyAllWindows()
            return closing


def opening(image):
    src = image
    cv.namedWindow("opening", cv.WINDOW_NORMAL)
    cv.createTrackbar("kernel shape",
                      "opening", 0, 2, lambda tmp: None)
    cv.createTrackbar("kernel size",
                      "opening", 0, 100, lambda tmp: None)
    while (True):
        kernel_size = cv.getTrackbarPos(
            "kernel size", "opening")
        kernel_shape = morph_shape(cv.getTrackbarPos(
            "kernel shape", "opening"))
        element = cv.getStructuringElement(kernel_shape, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                           (kernel_size, kernel_size))
        opening = cv.morphologyEx(src, cv.MORPH_OPEN, element)
        cv.imshow("opening", opening)
        if cv.waitKey(1) & 0xFF == ord("w"):
            cv.destroyAllWindows()
            return opening


def erode(image):
    src = image
    cv.namedWindow("erode", cv.WINDOW_NORMAL)
    cv.createTrackbar("kernel shape",
                      "erode", 0, 2, lambda tmp: None)
    cv.createTrackbar("kernel size",
                      "erode", 0, 100, lambda tmp: None)
    while (True):
        kernel_size = cv.getTrackbarPos(
            "kernel size", "erode")
        kernel_shape = morph_shape(cv.getTrackbarPos(
            "kernel shape", "erode"))
        element = cv.getStructuringElement(kernel_shape, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                           (kernel_size, kernel_size))
        erode = cv.erode(src, element)
        cv.imshow("erode", erode)
        if cv.waitKey(1) & 0xFF == ord("w"):
            cv.destroyAllWindows()
            return erode


def dilate(image):
    src = image
    cv.namedWindow("dilate", cv.WINDOW_NORMAL)
    cv.createTrackbar("kernel shape",
                      "dilate", 0, 2, lambda tmp: None)
    cv.createTrackbar("kernel size",
                      "dilate", 0, 100, lambda tmp: None)
    while (True):
        kernel_size = cv.getTrackbarPos(
            "kernel size", "dilate")
        kernel_shape = morph_shape(cv.getTrackbarPos(
            "kernel shape", "dilate"))
        element = cv.getStructuringElement(kernel_shape, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                           (kernel_size, kernel_size))
        dilate = cv.dilate(src, element)
        cv.imshow("dilate", dilate)
        if cv.waitKey(1) & 0xFF == ord("w"):
            cv.destroyAllWindows()
            return dilate
        
def gaussian(image):
    src = image
    cv.namedWindow("gaussian", cv.WINDOW_NORMAL)
    cv.createTrackbar("kernel size",
                      "gaussian", 3, 100, lambda tmp: None)
    while (True):
        kernel_size = cv.getTrackbarPos(
            "kernel size", "gaussian")
        if kernel_size % 2 == 0:
            kernel_size += 1
        gaussian = cv.GaussianBlur(src, (kernel_size, kernel_size), 0)
        cv.imshow("dilate", gaussian)
        if cv.waitKey(1) & 0xFF == ord("w"):
            cv.destroyAllWindows()
            return gaussian