# ObjectDetection 프로젝트

## 프로젝트 소개
- VGGNet 구조를 활용해서 이미지로부터 객체(object)를 검출(detection)하는 모델의 성능을 높이는 프로젝트를 진행했습니다.

## 프로젝트 기간
- 2023.03 ~ 2023.06

## 사용한 기법
- VOC2012 데이터
- VGGNet 백본 모델
- SSD 모델

## 파일 설명
- SSD_training.py 파일에서 전체적인 학습을 진행하여 가중치을 추출해냅니다.
- utils 폴더에는 SSD_training.py에서 사용할 함수를 모아놓은 디렉토리입니다. (미리 구현된 코드를 활용했습니다.)
- data_augmentation.py 파일은 증강기법(augmentation)을 구현해놓은 파일입니다.
- match.py 파일은 실제 객체 box와 모델이 예측한 box를 매칭하는 함수를 구현한 파일입니다.
- ssd_model.py, ssd_model_modified.py, ssd_predict_show.py 파일은 ssd 모델을 사용하기 위한 함수들을 모아놓은 파일입니다.

## 배운 점
- 처음 해보는 딥러닝 프로젝트여서 컴퓨터 비전 모델들의 특징과 구조를 공부해볼 수 있는 프로젝트였다
- 딥러닝 모델 학습의 전체적인 파이프라인과 사용되는 기법들을 이해하고 공부해볼 수 있는 프로젝트였다.
