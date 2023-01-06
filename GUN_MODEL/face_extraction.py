from facenet_pytorch import MTCNN
import os
import pandas as pd
import tqdm
import cv2
import torch


def face_extraction():
    DATA_FOLDER = './dataset'
    DATA_FOLDER_FAKE = './dataset/fake'
    DATA_FOLDER_REAL = './dataset/real'
    fake_videos = pd.DataFrame(
        list(os.listdir(DATA_FOLDER_FAKE)), columns=['Video'])
    real_videos = pd.DataFrame(
        list(os.listdir(DATA_FOLDER_REAL)), columns=['Video'])
    frames = []
    label = []
    for e in fake_videos['Video']:
        label.append('fake')
    fake_videos['Label'] = label
    
    label2 = []
    for e in real_videos['Video']:
        label2.append('real')
    real_videos['Label'] = label2
    
    train_videos = pd.concat([fake_videos, real_videos],
                             join='outer', ignore_index=True)
    print(train_videos.head)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=256, margin=40, select_largest=True,
                  post_process=False, device=device)
    for i, video_file in enumerate(train_videos['Video']):
        label = train_videos.loc[train_videos.Video == video_file, 'Label'].values[0]
        if label == 'fake':
            video_folder = DATA_FOLDER_FAKE
        elif label == 'real':
            video_folder = DATA_FOLDER_REAL
        video_path = video_folder+'/'+video_file
        capture_video = cv2.VideoCapture(video_path)
        video_len = int(capture_video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture_video.get(cv2.CAP_PROP_FPS)
        for frame_num in tqdm.tqdm(range(video_len)):
            success = capture_video.grab()
            if frame_num % fps == 0: # The frame between the fps the face almost not change so just ignore them
                 # Load frame
                success = capture_video.grab()
                success, frame = capture_video.retrieve()
            else:
                continue
            if not success:
                continue
            if frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect faces in batch
            if label == 'fake':
                save_paths = [
                    './dataset/crop/fake/'+f'video_{i}'+f'_{frame_num}.jpg']
            elif label == 'real':
                save_paths = [
                    './dataset/crop/real/'+f'video_{i}'+f'_{frame_num}.jpg']
    
            mtcnn(frame, save_path=save_paths)
     
        

if __name__ == '__main__':
    face_extraction()
