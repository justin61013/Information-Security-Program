import torch
from torch.utils.model_zoo import load_url
import cv2
from scipy.special import expit
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet,weights
from isplutils import utils

import time
from PIL import Image
import time
"""
Choose an architecture between
- EfficientNetB4
- EfficientNetB4ST
- EfficientNetAutoAttB4
- EfficientNetAutoAttB4ST
- Xception
"""
net_model = 'EfficientNetAutoAttB4'

"""
Choose a training dataset between
- DFDC
- FFPP
"""
train_db = 'DFDC'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
abc='111'
print(abc)
#model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
#print(model_url)
net = getattr(fornet,net_model)().eval().to(device)
net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'EfficientNetAutoAttB4_DFDC_bestval-72ed969b2a395fffe11a0d5bf0a635e7260ba2588c28683630d97ff7153389fc.pth')))
transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

facedet = BlazeFace().to(device).eval()
facedet.load_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)),"blazeface/blazeface.pth"))
facedet.load_anchors(os.path.join(os.path.dirname(os.path.realpath(__file__)),"./blazeface/anchors.npy"))
face_extractor = FaceExtractor(facedet=facedet)

def detect_face(image):
  #im_real = Image.open(image_path)
  im_real=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
  im_real_faces = face_extractor.process_image(img=im_real)
  #im_real_faces = face_extractor.process_image(img=image_path)

  if len(im_real_faces['faces'])==0:
    return None
    
  im_real_face = im_real_faces['faces'][0]
  faces_t = torch.stack( [ transf(image=im)['image'] for im in [im_real_face] ] )
  s1=time.time()
  with torch.no_grad():
    faces_pred = torch.sigmoid(net(faces_t.to(device))).cpu().numpy().flatten()
  s2=time.time()
  #print('pred',s2-s1)
  return faces_pred[0]

if __name__=='__main__':
  cap = cv2.VideoCapture( r'C:\Users\yaoching\Downloads\走廊-蔡佳吟-glasses+走廊-陳信杰-glasses.mp4')
  while(cap.isOpened()):
    ret, img = cap.read()
    print(detect_face(img))

