# 패키지 import
import os.path as osp
import random
import time
import tqdm

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data

from utils.ssd_model import SSD
from utils.ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn
from utils.ssd_model import MultiBoxLoss



# 1. 난수 시드 설정
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("사용 중인 장치:", device)


# 2. Dataset과 DataLoader를 작성한다

# 파일 경로 리스트를 취득
rootpath = "C:/Users/User/Desktop/2023-1/DeepLearning/objectdetection/data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath)

# Dataset 작성
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
color_mean = (104, 117, 123)  # (BGR) 색의 평균값
input_size = 300  # 이미지의 input 크기를 300×300으로 설정

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))


# DataLoader를 작성
batch_size = 32

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)

# 사전 오브젝트로 정리
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}



# 3. 네트워크 모델을 작성한다

# SSD300 설정
ssd_cfg = {
    'num_classes': 21,  # 배경 클래스를 포함한 총 클래스 수
    'input_size': 300,  # 이미지의 입력 크기
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 출력할 DBox의 화면비의 종류
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 각 source의 이미지 크기
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOX의 크기(최소)
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOX의 크기(최대)
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}


# SSD 네트워크 모델
net = SSD(phase="train", cfg=ssd_cfg)


num_epochs = 280 #에폭 수정

#10 epoch 학습 후 가중치 파일이 존재하는 경우, 'ssd300_10.pth'을 읽어 들여 model의 가중치를 업데이트함
#즉, 10 epoch부터 학습을 num_epochs 동안 수행함
#Modification by Young
start_epoch = 10 
weight_file = f'C:/Users/User/Desktop/2023-1/DeepLearning/objectdetection/weights/ssd300_{start_epoch}.pth'

# ssd의 기타 네트워크의 가중치는 He의 초기치로 초기화
def weights_init(m):
    if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
    #if m.bias is not None:  # 바이어스 항이 있는 경우
    #    nn.init.constant_(m.bias, 0.0)
    #Modification by Young
    for name, param in m.named_parameters():
        if 'bias' in name:  # 바이어스 항이 있는 경우, bias의 가중치를 0.0으로 설정
            nn.init.constant_(param, 0.0)
        

# loading pretrained weights for learning the SSD network
#Modification by Young
if osp.isfile(weight_file):  
    #start_epoch부터 end_epoch까지 학습 수행
    end_epoch = start_epoch + num_epochs
    # SSD의 학습된 가중치를 설정
    net_weights = torch.load( weight_file, map_location={'cuda:0': 'cpu'})
    net.load_state_dict(net_weights)
    print('네트워크 설정 완료: 학습된 가중치를 로드했습니다')
else:
    #1 epoch부터 end_epoch까지 학습수행, 즉 처음부터 학습을 수행함
    start_epoch = 0
    end_epoch = num_epochs
    # SSD의 초기 가중치를 설정
    # ssd의 vgg 부분에 가중치를 로드한다
    vgg_weights = torch.load('C:/Users/User/Desktop/2023-1/DeepLearning/objectdetection/weights/vgg16_reducedfc.pth')
    net.vgg.load_state_dict(vgg_weights)

    # He의 초기치를 적용
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    print('네트워크 설정 완료: 학습된 가중치를 로드했습니다')


# 4. 손실함수 및 최적화 기법의 설정

# 손실함수의 설정
criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

# 최적화 기법의 설정
optimizer = optim.SGD(net.parameters(), lr=1e-3,   #1e-3 -> 1e-3 로 수정 
                      momentum=0.9, weight_decay=5e-2)  #5e-4에서 5e-2로 수정


# 5. 학습 및 검증을 실시

# 모델을 학습시키는 함수 작성
def train_model(net, dataloaders_dict, criterion, optimizer, start_epoch, end_epoch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 중인 장치:", device)
    # 네트워크를 GPU로
    net.to(device)

    # 네트워크가 어느 정도 고정되면, 고속화시킨다
    torch.backends.cudnn.benchmark = True

    # 반복자의 카운터 설정
    iteration = 1
    epoch_train_loss = 0.0  # epoch의 손실합
    epoch_val_loss = 0.0  # epoch의 손실합
    logs = []

    # epoch 루프
    for epoch in range(start_epoch,end_epoch+1):

        # 시작 시간을 저장
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, end_epoch))
        print('-------------')

        # epoch별 훈련 및 검증을 루프
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # 모델을 훈련모드로
                print('(train)')
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()   # 모델을 검증모드로
                    print('-------------')
                    print('(val)')
                else:
                    # 검증은 10번에 1번만 실시
                    continue

            # 데이터 로더에서 minibatch씩 꺼내 루프
            iterator = tqdm.tqdm(dataloaders_dict[phase]) # ➊ 학습 로그 출력
            #for images, targets in dataloaders_dict[phase]:
            for images, targets in iterator:
                # GPU를 사용할 수 있으면, GPU에 데이터를 보낸다
                images = images.to(device)
                targets = [ann.to(device)
                           for ann in targets]  # 리스트의 각 요소의 텐서를 GPU로

                if phase == 'train':
                    # optimizer를 초기화
                    optimizer.zero_grad()
                    
                    # 'with torch.set_grad_enabled()' 문장을  validation시나 inference 시에 
                    # with torch.no_grad()을 도입하여 최신 pytorch 문법에 맞게 수정함
                    #with torch.set_grad_enabled(phase == 'train'):
                    
                    # 순전파(forward) 계산
                    outputs = net(images)

                    # 손실 계산
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    # 훈련시에는 역전파(Backpropagation)
                
                    loss.backward()  # 경사 계산

                    # 경사가 너무 커지면 계산이 불안정해지므로, clip에서 최대라도 경사 2.0에 고정
                    nn.utils.clip_grad_value_(
                        net.parameters(), clip_value=2.0)

                    optimizer.step()  # 파라미터 갱신

                    # ❷ tqdm이 출력할 문자열
                    #iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")
                    #if (iteration % 10 == 0):  # 10iter에 한 번, loss를 표시
                    t_iter_finish = time.time()
                    duration = t_iter_finish - t_iter_start
                        #print('반복 {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                        #    iteration, loss.item(), duration))
                    iterator.set_description('반복 {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                        iteration, loss.item(), duration))
                    t_iter_start = time.time()

                    epoch_train_loss += loss.item()
                    iteration += 1

                # 검증시
                else:
                    with torch.no_grad():
                        # 순전파(forward) 계산
                        outputs = net(images)

                        # 손실 계산
                        loss_l, loss_c = criterion(outputs, targets)
                        loss = loss_l + loss_c
                        epoch_val_loss += loss.item()

        # epoch의 phase 당 loss와 정답률
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss, epoch_val_loss))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # 로그를 저장
        log_epoch = {'epoch': epoch+1,
                     'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        epoch_train_loss = 0.0  # epoch의 손실합
        epoch_val_loss = 0.0  # epoch의 손실합

        # 네트워크를 저장한다
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(),  'C:/Users/User/Desktop/2023-1/DeepLearning/objectdetection/weights/ssd300_' +
                       str(epoch+1) + '.pth')


# 학습 및 검증 실시
#num_epochs= 250  

train_model(net, dataloaders_dict, criterion, optimizer, start_epoch, end_epoch)
