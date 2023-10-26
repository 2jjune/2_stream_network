import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import model
import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 10
learning_rate = 0.001

def default_loader(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames


# 데이터 전처리
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 데이터셋 불러오기
data_dir = "동영상 데이터 폴더 경로"

# video_files = glob.glob(os.path.join(data_dir, "*.mp4"))
#
# # 데이터셋을 train과 validation으로 나누기
# train_ratio=0.8
# num_train = int(len(video_files) * train_ratio)
# train_files = video_files[:num_train]
# val_files = video_files[num_train:]

# train_dataset = datasets.DatasetFolder(os.path.join(data_dir, "train"),
#                                         transform=transform1,
#                                         loader=default_loader,
#                                         extensions=".mp4",
#                                         classes=["nonfight", "fight"])
#
# val_dataset = datasets.DatasetFolder(os.path.join(data_dir, "val"),
#                                       transform=transform2,
#                                       loader=default_loader,
#                                       extensions=".mp4",
#                                       classes=["nonfight", "fight"])


train_dataset = load_dataset.VideoDataset(os.path.join(data_dir, 'train'), transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))
val_dataset = load_dataset.VideoDataset(os.path.join(data_dir, 'val'), transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 네트워크 및 optimizer 정의
net1 = model.WResNet50().to(device)
# net1 = model.ResNet50()
net2 = model.LSTMAE().to(device)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate)

# 학습 시작
for epoch in range(num_epochs):
    net1.train()
    net2.train()
    running_loss1 = 0.0
    running_loss2 = 0.0
    for i, data in enumerate(train_loader, 0):
        # 네트워크1 학습
        print(np.array(data).shape)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer1.zero_grad()
        outputs = net1(inputs)
        loss1 = criterion1(outputs, labels)
        loss1.backward()
        optimizer1.step()
        running_loss1 += loss1.item()

        # 네트워크2 학습
        inputs2 = []
        for frame in inputs:
            frame = np.asarray(frame.permute(1, 2, 0).numpy() * 255, dtype=np.uint8)
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            stds = [np.std(frame[..., i]) for i in range(frame.shape[-1])]
            # inputs2.append(np.std(frame_gray))
            inputs2.append(stds)
        inputs2 = np.asarray(inputs2).reshape(-1, 10)
        inputs2 = torch.from_numpy(inputs2).float().to(device)
        inputs2 = inputs2.unsqueeze(1)
        optimizer2.zero_grad()
        outputs2 = net2(inputs2)
        loss2 = criterion2(outputs2, inputs2)
        loss2.backward()
        optimizer2.step()
        running_loss2 += loss2.item()

    # 에폭마다 로스 출력
    print(f"Epoch {epoch+1} loss1: {running_loss1 / len(train_loader)}, loss2: {running_loss2 / len(train_loader)}")
    # validation
    net1.eval()
    net2.eval()
    val_loss1 = 0.0
    val_loss2 = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            # 네트워크1 검증
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net1(inputs)
            val_loss1 += criterion1(outputs, labels).item()

            # 네트워크2 검증
            inputs2 = []
            for frame in inputs:
                frame = np.asarray(frame.permute(1, 2, 0).numpy() * 255, dtype=np.uint8)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                inputs2.append(np.std(frame_gray))
            inputs2 = np.asarray(inputs2).reshape(-1, 10)
            inputs2 = torch.from_numpy(inputs2).float()
            inputs2 = inputs2.unsqueeze(1)
            inputs2 = inputs2.to(device)
            outputs2 = net2(inputs2)
            val_loss2 += criterion2(outputs2, inputs2).item()

    print(f"Validation Loss1: {val_loss1 / len(val_loader)}, Validation Loss2: {val_loss2 / len(val_loader)}")

print("Finished Training")

"""
# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 500
learning_rate = 0.001

# 데이터 전처리
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 네트워크 및 optimizer 정의
net1 = resnet50()
net2 = LSTMAE()
optimizer = optim.Adam(list(net1.parameters()) + list(net2.parameters()), lr=learning_rate)

# 비디오 프레임 처리
frames = []
labels = []
video_dirs = ['fight', 'nonfight']
for label, video_dir in enumerate(video_dirs):
    video_files = os.listdir(video_dir)
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                labels.append(label)
            else:
                break
        cap.release()

# 네트워크1과 네트워크2 입력 데이터 처리
inputs1 = []
inputs2 = []
num_frames = len(frames)
interval = num_frames // 10
for i in range(0, num_frames, interval):
    # 네트워크1 입력
    img = transform1(frames[i])
    inputs1.append(img)

    # 네트워크2 입력
    if i+10 <= num_frames:
        group = frames[i:i+10]
    else:
        group = frames[num_frames-10:num_frames]
    group = np.asarray(group)
    group = np.std(group, axis=0)
    img = transform2(group)
    inputs2.append(img)

inputs1 = torch.stack(inputs1)
inputs2 = torch.stack(inputs2)
labels = torch.tensor(labels)

# 학습
for epoch in range(num_epochs):
    for i in range(0, num_frames, batch_size):
        inputs1_batch = inputs1[i:i+batch_size]
        inputs2_batch = inputs2[i:i+batch_size]
        labels_batch = labels[i:i+batch_size]

        optimizer.zero_grad()

        outputs1 = net1(inputs1_batch)
        outputs2 = net2(inputs2_batch)

        loss1 = nn.CrossEntropyLoss()(outputs1, labels_batch)
        loss2 = nn.MSELoss()(outputs2, inputs2_batch)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{num_frames}], Loss: {loss.item()}")
"""