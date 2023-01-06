"""
**********************************************************************************************************
This program was developed by Kittiwat Pheramunchai and Shi-Jinn Horng in August 27, 2021, 
in the Information Security and Parallel Processing Laboratory which is located at 
National Taiwan University of Science and Technology. Do not copy of distribute it without permission.
**********************************************************************************************************
"""
from detect_from_video import test_full_image_network

PATH = r"C:\Thesis\Dora\Dataset\LAB_dataset\video\fake\走廊-尤耀慶-glasses+走廊-李冠霆-glasses.mp4"

test_full_image_network(PATH, write=False, cuda=False)