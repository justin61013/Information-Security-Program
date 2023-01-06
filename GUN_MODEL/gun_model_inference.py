import cv2
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch
from elaInceptionResnet import ElaInceptionResnet
import torch.nn as nn
from PIL import Image
import numpy as np

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

def fake_inference(image, model,cuda=True):
    #image:cv2 format
    mtcnn = MTCNN(image_size=256, margin=40, select_largest=True, post_process=False).eval()
    face = mtcnn(image)
    if face[0] == None:
        return None
    img_pil = transforms.ToPILImage(mode='RGB')(face[0].squeeze_(0))
    open_cv_image = np.array(img_pil)
    prediction, output = predict_with_model(open_cv_image, model, cuda=cuda)
    return prediction

if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ElaInceptionResnet(classify=True, pretrained=True,device=device,weight_path=r'C:\Users\yaoching\Desktop\Face-recognition-attcak-defense\FACE_DATA_COLLETCOR\GUN_MODEL\weight\lab_no.pt').to(device)
    model.eval()
    cap = cv2.VideoCapture( r'C:\Users\yaoching\Downloads\陳信杰.mp4')
    while(cap.isOpened()):
        ret, img = cap.read()
        print(fake_inference(img,model,True))