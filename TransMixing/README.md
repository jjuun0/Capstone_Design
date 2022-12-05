

### 리눅스 세팅

- Ubuntu 20.04, CUDA 11.3, cuDNN 8.2.0, PyTorch 1.11.0, Python 3.7

- 가상환경 만들기 및 라이브러리 설치
  
  - `conda create --name transmixing python=3.7` 
  - `conda install -c conda-forge tqdm`
  - `conda install -c conda-forge gdown`
  - `conda install -c anaconda scikit-learn=0.22`
  - `conda install -c anaconda scipy` 
  - `conda install -c conda-forge lpips`
  - `conda install -c conda-forge dlib` 
  - `conda install -c conda-forge wandb` `conda install -c conda-forge matplotlib`

- PyTorch 설치
  
  - `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

- ninja version 바꾸기
  
  - `!wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip`
  
  - `!sudo unzip ninja-linux.zip -d /usr/local/bin/`

- 본인 가상환경의 bin 폴더 경로 수정해야함
  
  - `!sudo update-alternatives --install /home/project/anaconda3/envs/jojo/bin/ninja ninja /usr/local/bin/ninja 1 --force`



### 코드 설명

- `demo_flask/`: 데모

- `gan_inversion/` : E4E, II2S, Restyle 모델의 Gan Inversion

- `images/`: Style, Input 이미지들

- `losses/`: 학습에 사용된 loss

- `op/`: StyleGAN2에 필요한 모듈

- `style_mixing/`: 선형, 비선형 스타일 결합

- `config.json, config_parser.py`: 학습에 사용하는 config

- `ffhq_align.py` : 이미지의 얼굴 모양 정렬

- `train_transmixing.py, predict_transmixing.py`: 학습, 예측



### 사용법

- `config.json`을 수정함

- `python train_transmixing.py`로 모델 학습

- `python predice_transmixing.py`로 모델 추론



### 참고 레퍼런스

- One-shot generation & Style Transfer
  
  - [JoJoGAN](https://github.com/mchong6/JoJoGAN)

- GAN Inversion
  
  - [ii2s](https://github.com/ZPdesu/II2S)
  
  - [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder)
