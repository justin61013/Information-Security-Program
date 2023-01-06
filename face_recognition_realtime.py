from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
from PIL import ImageDraw,ImageFont
import cv2
import numpy
import torch
import os
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import math
import serial
import serial.tools.list_ports
import time
import _thread
import sys
import argparse
from attb4 import face_detect

#mtcnn 解析度不能太高
#持續顯示pass or failed
def normal_dispaly(mtcnn,resnet,predictor,gallery_emb,gallery_map,is_pass):
    count=0
    while (True):
        #ret為是否成功，frame為cv2畫面
        ret, frame = cap.read()      #读取图像并显示

        #方便後面dlib調用
        frame_cv2=frame
        #轉為PIL圖片
        frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        #只輸出最大信心度的臉，以及坐標位置和信心度
        img_cropped ,batch_boxes,batch_probs= mtcnn(frame) 

        draw=ImageDraw.ImageDraw(frame)

        #如果有偵測到人臉的話
        if batch_boxes is not None:
            #將剛才mtcc切出來的人臉(tensor)，丟給resnet
            img_embedding = resnet(img_cropped.unsqueeze(0).cuda())
            #定義一個dlib.rect方便輸入給後續的dlib形狀偵測器
            box=dlib.rectangle(int(batch_boxes[0][0]),int(batch_boxes[0][1]),int(batch_boxes[0][2]),int(batch_boxes[0][3]))

            #shape:各個點坐標的ndarray:68x2
            shape = predictor(frame_cv2, box)
            shape = face_utils.shape_to_np(shape)

            #計算該照片的向量和每個gallery向量的距離
            distance=[]
            for v in range(gallery_emb.shape[0]):
                d=np.sqrt(np.sum(np.square(img_embedding.cpu().detach().numpy()-gallery_emb[v,:])))
                distance.append(d)

            print(gallery_map[distance.index(min(distance))],'face distance：',min(distance))
            
            draw.rectangle((batch_boxes[0][0],batch_boxes[0][1],batch_boxes[0][2],batch_boxes[0][3]),outline ='red',width =5)

            #如果向量跟gallery中任一一張最小距離小於0.7，則顯示該張照片的名字，並acess,否則顯示denied
            if(min(distance)<0.7):
                draw.text((int(batch_boxes[0][0]),int(batch_boxes[0][1])-30),gallery_map[distance.index(min(distance))], font=fontStyle, fill=(255,0,0))
            else:
                draw.text((int(batch_boxes[0][0]),int(batch_boxes[0][1])-30),r'denied', font=fontStyle, fill=(255,0,0))
            if is_pass==True:
                draw.text((5,5),'PASS!', font=fontStyle, fill=(0,255,0))
            else:
                draw.text((5,5),'Failed!', font=fontStyle, fill=(255,0,0))
                #------------只顯示5幀Failed------------
                if count==5:
                    return 0
            #轉換回cv2格式
            frame = cv2.cvtColor(numpy.asarray(frame),cv2.COLOR_RGB2BGR)

            #將dlib獲得的人臉關鍵點畫上去
            #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


        else:
            if is_pass==True:
                draw.text((5,5),'PASS!', font=fontStyle, fill=(0,255,0))
            else:
                draw.text((5,5),'Failed!', font=fontStyle, fill=(255,0,0))
                #------------只顯示5幀Failed------------
                if count==5:
                    return 0
            frame = cv2.cvtColor(numpy.asarray(frame),cv2.COLOR_RGB2BGR)
        
        #從PIL轉回cv2並顯示
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #--------------持續顯示75幀PASS------------
        count+=1
        if count==45:
            return 0

#臉部不同特征點組成線段的比例
def turn_aspect_ratio(x1,x2,x3):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(x1, x2)
	B = dist.euclidean(x2, x3)

	return A/B

#檢測此時的人是否是活人,如果此時人脸多幀miss，或發現換人，同樣會Failed
def liveness_detect(mtcnn,resnet,predictor,gallery_emb,gallery_map,index):

    #防止中間某一幀沒偵測到就直接退出
    miss=0
    #防止换人来试图通过活体检测
    person_wrong=0
    count=0
    #用來判斷是否有搖頭
    last_ratio=None
    abs_ratio=0
    fake_value=0
    while (True):
        ret, frame = cap.read()
        is_fake=face_detect.detect_face(frame)
        frame_cv2=frame
        frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        draw=ImageDraw.ImageDraw(frame)
        if is_fake is not None:
            if is_fake>0.1 and is_fake<0.25:
                draw.text((300,5),'Maybe Fake', font=fontStyle, fill=(255,0,0))
                fake_value+=0.5
            elif is_fake<0.1:
                draw.text((300,5),'Real', font=fontStyle, fill=(255,0,0))
            else:
                draw.text((300,5),'Fake', font=fontStyle, fill=(255,0,0))
                fake_value+=1
        draw.text((5,5),'請輕微搖頭', font=fontStyle, fill=(255,0,0))
        if fake_value>cf.fake_threshold:
            print('Fake Face Failed!')
            return False
        #只輸出最大信心度的臉，以及坐標位置和信心度

        img_cropped ,batch_boxes,batch_probs= mtcnn(frame)

        #30幀沒偵測到人臉的話就失敗
        if batch_boxes is None:
            miss+=1
            draw.text((frame.size[0]-200,5),'<Person miss>', font=fontStyle, fill=(0,255,255))
            if(miss)>30:
                return False
            frame = cv2.cvtColor(numpy.asarray(frame),cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:

            #將剛才mtcc切出來的人臉(tensor)，丟給resnet
            img_embedding = resnet(img_cropped.unsqueeze(0).cuda())
            #定義一個dlib.rect方便輸入給後續的dlib形狀偵測器
            box=dlib.rectangle(int(batch_boxes[0][0]),int(batch_boxes[0][1]),int(batch_boxes[0][2]),int(batch_boxes[0][3]))

            #shape:各個點坐標的ndarray:68x2
            shape = predictor(frame_cv2, box)
            shape = face_utils.shape_to_np(shape)

            #計算和各個gallery照片的距離
            distance=[]
            for v in range(gallery_emb.shape[0]):
                d=np.sqrt(np.sum(np.square(img_embedding.cpu().detach().numpy()-gallery_emb[v,:])))
                distance.append(d)

            #如果連續多幀信心度變低或者發現換人了，都會Failed
            #unexpected bug:今天如果一個人有多張gallery，此處可能會出錯 new:又似乎不會
            if gallery_map[distance.index(min(distance))] !=gallery_map[index] or min(distance)>0.7:
                person_wrong+=1
                draw.text((frame.size[0]-260,5),'<Person changed>', font=fontStyle, fill=(255,255,0))
                if person_wrong>30:
                    return False
            else:
                #這個區塊確保存在人臉且是正確的人
                count+=1
                #計算累計ratio的變化量
                if last_ratio is None:
                    last_ratio = turn_aspect_ratio(shape[1],shape[28],shape[15])
                else:
                    #累積ratio超過1.5則return pass信息
                    abs_ratio=abs(last_ratio-turn_aspect_ratio(shape[1],shape[28],shape[15]))
                    if abs_ratio>cf.liveness_threshold:
                        return True

            #畫出偵測到的框和名字
            if min(distance)<0.7:
                draw.text((int(batch_boxes[0][0]),int(batch_boxes[0][1])-30),gallery_map[distance.index(min(distance))], font=fontStyle, fill=(255,0,0))
            else:
                draw.text((int(batch_boxes[0][0]),int(batch_boxes[0][1])-30),r'denied', font=fontStyle, fill=(255,0,0))
            draw.rectangle((batch_boxes[0][0],batch_boxes[0][1],batch_boxes[0][2],batch_boxes[0][3]),outline ='red',width =5)
            

            frame = cv2.cvtColor(numpy.asarray(frame),cv2.COLOR_RGB2BGR)

            #畫臉上的點
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            #需要在150幀內累積足夠的ratio
            if count==150:
                print('Failed!')
                return False

class UnNormalize(object): #為了可視化，反normailize
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
if __name__ == '__main__':

    #加載必要參數
    parser = argparse.ArgumentParser()

    # base argument
    parser.add_argument('--predictor', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'shape_predictor_68_face_landmarks.dat'),
                         help='weight path of predictor')

    parser.add_argument('--gallery', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'gallery'),
                         help=' path of face database')

    parser.add_argument('--font', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'simsun.ttc'),
                         help='path of font')

    parser.add_argument('--liveness-threshold', type=int, default=2.5, help='threshold for liveness,larger will be harder, as well time will be longer')

    parser.add_argument('--fake-threshold', type=int, default=13, help='threshold for fake face number')
    cf = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:',device)

    
    #創建模型
    mtcnn = MTCNN(margin=0,device=device,thresholds=[0.9,0.9,0.9]).eval()
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    #臉部特征點偵測器
    predictor = dlib.shape_predictor(cf.predictor)


    #計算各gallery樣本的embedding(gallery_emb)，記錄gallery中的人名(gallery_map)
    gallery_emb=[]
    gallery_map=[]

    #計算gallery資料夾中每個人臉的embedding以用於後續的比對
    for filename in os.listdir(cf.gallery):
        img = Image.open(os.path.join(cf.gallery,filename))
        img_cropped ,batch_boxes,probs= mtcnn(img)
        img_embedding = resnet(img_cropped.unsqueeze(0).cuda())
        gallery_emb.append(img_embedding[0].cpu().detach().numpy())
        gallery_map.append(filename.split('.')[0])

    #gallery的embedding向量shape：[number of images,embedding dimensions]
    #從tensor轉為numpy
    gallery_emb=numpy.array(gallery_emb)


    cap = cv2.VideoCapture(0)        #'0'选择笔记本电脑自带参数，‘1’为USB外置摄像头
    cap.set(propId=3, value=1280)     #设置你想捕获的视频的宽度
    cap.set(propId=4, value=960)     #设置你想捕获的视频的高度
    #print(cap.get(3), cap.get(4))    #验证是否设置成功
    #cap = cv2.VideoCapture( r'C:\Users\yaoching\Desktop\Face-recognition-attcak-defense\FACE_DATA_COLLETCOR\demo_data\TTest.mp4')
    #設置要顯示名字的中文字體
    fontStyle = ImageFont.truetype(cf.font, 30, encoding="utf-8")

    #假定门锁的状态
    status='lock'
    #不断获取摄影机画面
    while (True):
        #ret為是否成功，frame為cv2畫面
        ret, frame = cap.read()      #读取图像并显示

        #方便後面dlib調用
        frame_cv2=frame

        #轉為PIL圖片
        frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        #只輸出最大信心度的臉，以及坐標位置和信心度
        img_cropped ,batch_boxes,batch_probs= mtcnn(frame) 
        
        #如果有偵測到人臉的話
        if batch_boxes is not None:
            #將剛才mtcc切出來的人臉(tensor)，丟給resnet
            img_embedding = resnet(img_cropped.unsqueeze(0).cuda())
            #定義一個dlib.rect方便輸入給後續的dlib形狀偵測器
            box=dlib.rectangle(int(batch_boxes[0][0]),int(batch_boxes[0][1]),int(batch_boxes[0][2]),int(batch_boxes[0][3]))

            #shape:各個點坐標的ndarray:68x2
            shape = predictor(frame_cv2, box)
            shape = face_utils.shape_to_np(shape)

            #計算該照片的向量和每個gallery向量的距離
            distance=[]
            for v in range(gallery_emb.shape[0]):
                d=np.sqrt(np.sum(np.square(img_embedding.cpu().detach().numpy()-gallery_emb[v,:])))
                distance.append(d)

            print(gallery_map[distance.index(min(distance))],'face distance：',min(distance))
            
            #畫出偵測到的框和名字
            draw=ImageDraw.ImageDraw(frame)

            draw.rectangle((batch_boxes[0][0],batch_boxes[0][1],batch_boxes[0][2],batch_boxes[0][3]),outline ='red',width =5)

            #如果向量跟gallery中任一一張最小距離小於0.7，則顯示該張照片的名字，並acess,否則顯示denied
            if(min(distance)<0.7):
                draw.text((int(batch_boxes[0][0]),int(batch_boxes[0][1])-30),gallery_map[distance.index(min(distance))], font=fontStyle, fill=(255,0,0))
                #啟動活體檢測
                liveness=liveness_detect(mtcnn,resnet,predictor,gallery_emb,gallery_map,distance.index(min(distance)))
                if liveness== True:
                    #持續顯示一段時間pass
                    #開鎖
                    #_thread.start_new(unlock,())
                    
                    normal_dispaly(mtcnn,resnet,predictor,gallery_emb,gallery_map,is_pass=True)
                    continue
                else:
                    normal_dispaly(mtcnn,resnet,predictor,gallery_emb,gallery_map,is_pass=False)
                    continue
            else:
                draw.text((int(batch_boxes[0][0]),int(batch_boxes[0][1])-30),r'denied', font=fontStyle, fill=(255,0,0))
            frame = cv2.cvtColor(numpy.asarray(frame),cv2.COLOR_RGB2BGR)

            #將dlib獲得的人臉關鍵點畫上去
            #cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        else:
            frame = cv2.cvtColor(numpy.asarray(frame),cv2.COLOR_RGB2BGR)
        
        #從PIL轉回cv2並顯示
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()                    #按‘q’键退出后，释放摄像头资源
    cv2.destroyAllWindows()




