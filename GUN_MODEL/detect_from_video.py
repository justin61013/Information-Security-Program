"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -o <path to output folder, will write one or multiple output videos there>

**********************************************************************************************************
This program was developed by Kittiwat Pheramunchai and Shi-Jinn Horng in August 27, 2021, 
in the Information Security and Parallel Processing Laboratory which is located at 
National Taiwan University of Science and Technology. Do not copy of distribute it without permission.
**********************************************************************************************************
"""
import os
import argparse
from os.path import join
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import MTCNN
from elaInceptionResnet import ElaInceptionResnet


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("image",image)
    # cv2.waitKey()
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = transforms.Compose([
                    transforms.Resize(256),
                    np.float32,
                    transforms.ToTensor(),
                ])
    preprocessed_image = preprocess(Image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (0 = fake, 1 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def test_full_image_network(video_path, write=False, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param write: enable save the result video to the result folder at the same location as the original video
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'_detected.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    mtcnn = MTCNN(image_size=256, margin=40, select_largest=True, post_process=False)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ElaInceptionResnet(classify=True, pretrained=True, num_classes=2,device=device).to(device)
    model.eval()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video

    for frame_num in tqdm(range(num_frames)):
        success = reader.grab()
        if frame_num % fps == 0:
            _, image = reader.read()
        else:
            continue

        if not success:
            continue

        if image is None:
            break

        # Image size
        height, width = image.shape[:2]

        # Init output writer
        if write == True:
            if writer is None:
                writer = cv2.VideoWriter("./result"+video_fn, fourcc, fps,(height, width)[::-1])

        face = mtcnn(image)
        if face == None:
            print("Can't detect face")
            continue
        if len(face):
            img_pil = transforms.ToPILImage(mode='RGB')(face.squeeze_(0))
            # img_pil.show()
            open_cv_image = np.array(img_pil)

            # Actual prediction using our model
            prediction, output = predict_with_model(open_cv_image, model, cuda=cuda)
            print(prediction)
            # ------------------------------------------------------------------
            # Draw boxes and save faces
            boxes, probs = mtcnn.detect(image)
            img_draw = image.copy()

            # Text and bb
            x = int(boxes[0].tolist()[0])
            y = int(boxes[0].tolist()[1])
            w = int(boxes[0].tolist()[2]) - x
            h = int(boxes[0].tolist()[3]) - y
            label = 'fake' if prediction == 0 else 'real'
            color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]
            cv2.putText(img_draw, str(output_list)+'=>'+label, (x, y+h+30),
                        font_face, font_scale,
                        color, thickness, 2)
            # draw box over face
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 2)

        # Show
        cv2.imshow('test', img_draw)
        cv2.waitKey(5)     # Show frame for 5 sec
        if write == True:
            writer.write(img_draw)
    if writer is not None:
        writer.release()

def test_image_network(image_path, cuda=True):
    image = cv2.imread(image_path)
    mtcnn = MTCNN(image_size=256, margin=40, select_largest=True, post_process=False)
    face = mtcnn(image)
    img_pil = transforms.ToPILImage(mode='RGB')(face.squeeze_(0))
    open_cv_image = np.array(img_pil)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ElaInceptionResnet(classify=True, pretrained=True, num_classes=2,device=device).to(device)
    model.eval()
    prediction, output = predict_with_model(open_cv_image, model, cuda=cuda)
    return prediction

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--write', action='store_true',
                   default='.')
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_full_image_network(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            test_full_image_network(**vars(args))
