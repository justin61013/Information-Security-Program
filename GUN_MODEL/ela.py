"""
**********************************************************************************************************
This program was developed by Kittiwat Pheramunchai and Shi-Jinn Horng in August 27, 2021, 
in the Information Security and Parallel Processing Laboratory which is located at 
National Taiwan University of Science and Technology. Do not copy of distribute it without permission.
**********************************************************************************************************
"""
from PIL import Image, ImageChops, ImageEnhance
import os, cv2
import numpy as np


def ela(image):
    """
    Generates an ELA image

    """

    im = image
    im.save("./temp.jpg", 'JPEG', quality=92)

    tmp_fname_im = Image.open("./temp.jpg")
    ela_im = ImageChops.difference(im, tmp_fname_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    os.remove("./temp.jpg")
    # ela_im.show()
    open_cv_image = cv2.cvtColor(np.array(ela_im), cv2.COLOR_RGB2BGR)

    return open_cv_image


if __name__ == '__main__':
    img = Image.open(r"C:\Thesis\FaceForensics-master\FaceForensics-master\classification\Kittiwat_dataset\dataset\crop\demo\fake\video_3whiten_frame_150.jpg")
    ela(img)
