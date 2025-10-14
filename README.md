사전에 생성한 로그 데이터를 기반으로 AutoEncoder 모델로 학습 후, 정상패턴을 압축 / 복원하고, 재구성 오차를 통해 Anomaly를 탐지한 프로젝트입니다.
데이터 생성하고 전처리하여 CPU 환경에서 실험하였습니다.

1. Python 최신버젼이 필요하며 추가적으로 필요한 사항은 requirements.txt에 저장해두었습니다.
   
2. 윈도우 환경에서 cmd창을 열고 python -m venv <테스트 폴더명>
  venv/scripts/activate 명령으로 가상환경을 만들어줍니다.
  pip install -r requirements.txt 명령으로 필요한 요소를 다운로드 합니다.

4. python data_gen.py --out logs.csv --n <생성할_로그의_숫자> 명령으로 테스트 할 로그파일 생성해줍니다.
  
5. python preprocess.py --in logs.csv --out-prefix features 명령으로 전처리 후 파일을 저장합니다.
  (features는 하드코딩한 파일명이라 수정할 경우 preprocess.py도 같이 수정바랍니다.)

6. python train.py --features features.npy --labels features_labels.npy --epochs <반복횟수> 명령으로 원하는 만큼 전체 데이터를 학습합니다.
  학습완료 시 checkpoints/ae.pth를 생성합니다. 
   
7. python evaluate.py --model checkpoints/ae.pth --features features.npy --labels features_labels.npy
  python inference_demo.py --model checkpoints/ae.pth 명령으로 학습이 끝난 ae.pth파일을 불러와 테스트용 feature 데이터를 입력합니다.
  재구성 오차를 계산 후 간단하게 시각화 해줍니다. 


