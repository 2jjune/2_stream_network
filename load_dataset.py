import os
import random
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from torchvision import transforms
from PIL import Image


random.seed(42)
transformss = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 데이터셋 인스턴스 생성
class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=transformss):
        self.data_dir = data_dir
        self.transform = transform

        # 동영상 파일 목록
        self.files = []
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            for filename in os.listdir(label_dir):
                self.files.append((os.path.join(label_dir, filename), label))

        # 클래스 인덱스
        self.class_indices = {'fight': 0, 'nonfight': 1}
        # self.transform = transform if transform else self._get_transform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 동영상 파일
        # print("self.files: ", self.files)
        filepath, label = self.files[idx]

        # 동영상 프레임 로드
        cap = cv2.VideoCapture(filepath)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # 동영상 프레임을 이미지로 변환 후 네트워크 입력에 맞게 전처리
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]#동영상의 각 프레임마다 self.transform 함수가 적용되어 3차원 이미지 데이터가 1차원 Tensor로 변환
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        # 네트워크 입력으로 사용할 프레임 이미지 선택
        num_frames = len(frames)

        if num_frames < 10:
            # 프레임 수가 10개보다 작을 경우, 마지막 프레임을 복제하여 총 10개로 만듦
            # frames.extend([frames[-1]] * (10 - num_frames))
            frames = torch.cat([frames, frames[-1].unsqueeze(0).repeat(10 - num_frames, 1, 1, 1)])
        frame_indices = sorted(random.sample(range(num_frames), 10))
        # frames = frames[frame_indices]
        frames = [frames[i] for i in frame_indices]
        frames = torch.stack(frames)#frames를 다시 4차원 Tensor로 변환합니다. 이 때 torch.stack 함수는 여러 개의 3차원 Tensor를 받아서 4차원 Tensor로 쌓아올리므로, 다시 (960, 360, 640, 3)과 같은 4차원 shape로 돌아갑니다.

        # 클래스 레이블을 숫자로 변환하여 반환
        label_idx = self.class_indices[label]
        print('type of frames, label_idx-------------------------------------------------------------')
        print(type(frames), type(label_idx))
        return frames, label_idx
