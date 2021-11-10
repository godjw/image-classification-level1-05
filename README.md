# Image Classfication Level1-05 AIM 
[Information](https://github.com/godjw/image-classification-level1-05/blob/master/%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3_level1_5.pdf)

## Wandb Login

1. wandb_setting/wandb_login.py 실행 
2. Wandb Login 
3. Wandb에서 제공하는 API key  입력
- wandb_setting/wandb_config.josn 에서 project와 name value 값을 자유롭게 지정

## Installation

```sh
wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000074/data/train.tar.gz
tar -zxvf train.tar.gz
pip3 install -r requirements.txt
pip3 install upgrade torchvision
pip3 install wandb
sudo apt-get install libgl1-mesa-glx
```
## how to train model 

환경변수 SM_CHANNEL_TRAIN 에 train dataset images 경로 설정
```sh
SM_CHANNEL_TRAIN={directory}/train/images
```
환경변수 SM_MODEL_DIR 에 model 저장 경로 설정 
```sh
SM_MODEL_DIR=./{model}
```
train.py 실행
```sh
python train.py
```

## ensemble model

**mode** 인자를 사용해, age/gender/mask 3개의 모델로 나누어서 학습

### age model
```sh
python train.py --mode [age]
```
### mask model
age model의 디렉토리 경로와 동일하게 name을 설정
```sh
python train.py [mask] --dump true --name exp{number}
```
### gender model
age, mask model의 디렉토리 경로와 동일하게 name을 설정
```sh
python train.py [gender] --dump true --name exp{number}
```
## how to inference 

환경변수 SM_CHANNEL_EVAL에 eval 경로 설정
```sh
SM_CHANNEL_EVAL={directory}/eval
```
name 인자와 model_name을 이용해 모델 경로 모델 파일을 지정하여 inference.py 실행
```sh
python inference.py --name exp{number} --model_name {bestf1.pt|best.pt}
```
ensemble model inference의 경우 mode 인자 추가
```sh
python inference.py --mode ensemble --name exp{number}
```

