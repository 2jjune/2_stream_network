import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import random
import model
import load_dataset
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from torchsummary import summary
from torch.autograd import Variable
import torch.multiprocessing as multiprocessing
import torch.multiprocessing as mp
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 데이터셋 폴더 경로
    data_dir = 'D:/Jiwoon/dataset/two_stream_ubi_data/'

    # 하이퍼파라미터 설정
    batch_size = 1
    lr = 1e-4
    num_epochs = 500

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # 데이터셋 인스턴스 생성
    class VideoDataset(Dataset):
        def __init__(self, data_dir, transform=transform):
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

            # 네트워크 입력으로 사용할 프레임 이미지 선택
            num_frames = len(frames)

            if num_frames < 10:
                # 프레임 수가 10개보다 작을 경우, 마지막 프레임을 복제하여 총 10개로 만듦
                frames.extend([frames[-1]] * (10 - num_frames))
            frame_indices = sorted(random.sample(range(num_frames), 10))
            frames = [frames[i] for i in frame_indices]
            frames = torch.stack(frames)#frames를 다시 4차원 Tensor로 변환합니다. 이 때 torch.stack 함수는 여러 개의 3차원 Tensor를 받아서 4차원 Tensor로 쌓아올리므로, 다시 (960, 360, 640, 3)과 같은 4차원 shape로 돌아갑니다.

            # 클래스 레이블을 숫자로 변환하여 반환
            label_idx = self.class_indices[label]
            return frames, label_idx

    train_dataset = VideoDataset(data_dir+'train/', transform=transform)
    val_dataset = VideoDataset(data_dir+'val/', transform=transform)

    # 데이터셋 분할 (train:validation = 8:2)
    # num_train = int(len(dataset) * 0.8)
    # num_val = len(dataset) - num_train
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("--------------데이터 생성 완료---------------------")
    # Two-Stream Network 정의



    print('-+----------------network----------------')

    two_stream_network = model.TwoStreamNetwork(device).to(device)
    optimizer = optim.Adam(two_stream_network.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('-+----------------network over----------------')

    # 손실 함수 정의하기
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    # 학습하기
    for epoch in range(num_epochs):
        since = time.time()
        two_stream_network.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"---------{epoch+1}---------start")

        for i, (inputs, targets) in enumerate(train_loader):
            # gpu 연산을 위해 입력 데이터와 타깃을 cuda로 보내기
            # print(np.array(inputs).shape)
            targets = targets.view(-1,1)
            targets = targets.float()
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            # resnet_out, lstm_out = two_stream_network(inputs)
            fusion_out = two_stream_network(inputs)
            result = []
            for j in range(batch_size):
                if fusion_out[j].item()>0.7:
                    result.append(torch.tensor(1.0, device=device))
                else:
                    result.append(torch.tensor(0.0, device=device))
            result = torch.tensor(result, device=device)
            result = result.view(-1,1)
            # print(f'fusion out : {fusion_out.item()}, targets : {targets.item()}')
            # Loss 계산
            loss = criterion(fusion_out, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 통계 정보 업데이트
            running_loss += loss.item()
            # _, predicted = fusion_out.max(1)
            # _, predicted = result.max(1) # softmax용
            total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            correct += result.eq(targets).sum().item()
            # print(f'{i} complete')
            # print(f'result : {result}, targets:{targets}, total:{total}, correct:{correct}')
            #1이 nonfight 0은 fight
        # 학습 결과 출력
        print('[Epoch: %d] Train Loss: %.5f | Train Acc: %.3f%% (%d/%d)' %
              (epoch + 1, running_loss / len(train_loader), 100. * correct / total, correct, total))

        # Evaluation 과정
        two_stream_network.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        result_list = []
        targets_list = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                # gpu 연산을 위해 입력 데이터와 타깃을 cuda로 보내기
                targets = targets.view(-1, 1)
                targets = targets.float()
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                # resnet_out, lstm_out = two_stream_network(inputs)
                fusion_out = two_stream_network(inputs)
                result = []
                for j in range(batch_size):
                    if fusion_out[j].item() > 0.7:
                        result.append(torch.tensor(1.0, device=device))
                    else:
                        result.append(torch.tensor(0.0, device=device))

                    result_list.append(result[j].item())
                    targets_list.append(targets[j].item())
                result = torch.tensor(result, device=device)

                # Loss 계산
                result = result.view(-1,1)
                loss = criterion(fusion_out, targets)

                # 통계 정보 업데이트
                running_loss += loss.item()
                # _, predicted = fusion_out.max(1)
                # _, predicted = result.max(1)
                # print(result)
                total += targets.size(0)
                correct += result.eq(targets).sum().item()

        # 평가 결과 출력
        time_elapsed = time.time() - since
        print('[Epoch: %d] Val Loss: %.5f | Val Acc: %.3f%% (%d/%d)    (%.0fm %.0fs)' %
              (epoch + 1, running_loss / len(val_loader), 100. * correct / total, correct, total, time_elapsed // 60, time_elapsed % 60))
        print(f'            predict : {result_list}, 1=nonfight 0=fight')
        print(f'              label : {targets_list}, 1=nonfight 0=fight')
    print('Finished Training')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
    # multiprocessing.freeze_support()
    # freeze_support()
