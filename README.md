## SAM-Adapter Architecture Search

- 🦑5/27 
  - create repo ✅
  - create conda env ✅
  - write search.py ✅
  - search configs are set in search_demo.yaml ✅ 

- 🐔5/28
  - search.py > random architecture generation() ✅
  - search.py > prepare_training() 
  - search.py > compute_nas_score() 🔃 -> additional editing 
  - ./model/sam.py > Class Image encoder setting
  - ./model/mmseg/models/sam/image_encoder.py > Class Prompt generator setting


- 🐠6/2
  - search.py > random search architecture 에 대한 score list를 구축하는 방식으로 변경 ✅
  - search.py > prepare_training()
  - search.py > compute_nas_score()
  - ./model/sam.py > Class Image encoder setting
  - ./model/mmseg/models/sam/image_encoder.py > Class Prompt generator setting

- 🍥6/4
  - search.py > prepare_training() ✅
    - model의 search_config 가 설정된 대로 전달되도록 설정
  - search.py > inference를 compute_zico.py 에서 할 수 있도록 코드 수정 ✅
    - ./model/sam.py > Class SAM > search_backward() 추가: gradient만 역전파하고 모델의 update는 하지 않도록 설정
  - ./model/sam.py > Class Image encoder setting ✅
  - ./model/mmseg/models/sam/image_encoder.py > Class Prompt generator setting ✅
  - search.py > 실험 진행 과정 시각화 끝

- 🚀6/5
  - nas로 폴더 옮겨서 실험 진행 > svr3에서 진행해보기
  - CAMO 데이터셋으로 실험 진행 중 ✅
  - 실험 돌아감 > loss랑 nas score 출력됨
  - sam-b: 15152M 정도 차지함 ✅
  - nas_score 뿐만 아니라 loss에 대한 점수도 고려할 수 있는 파이프라인 구축 필요

- 6/7
  - 4만개의 sample을 모두 다루는 것은 말이 안됨
  - 우선은 1000개를 random sampling 하여 선택하는 것으로 결정

- 💦6/10
  - CAMO 데이터셋에 대한 search 종료 ✅
  - promise12 데이터셋 실험 가능하도록 구축 ✅
  - CAMO 데이터셋 train 진행 ✅
  - search.py 마지막에 저장하는 부분 오류 해결 ✅
  - train.py 수정 ✅
  - promise12 데이터셋 실험 진행
    - image 크기가 256*256 이기 떄문인지? 에러 발생
    -> 원래 sam은 1024 size에서만 구동된다 함 > 256*256에서는 사용이 안됨!!!
  - camelyon 데이터셋을 1024 patch로 잘라서 사용해보기

- ⚾6/11
  - sam-adapter의 vit-b 모델을 이용한 훈련
  - 실험 결과> original model이 약간 더 높게 나옴

- ☕6/12
  - polyp 데이터셋을 이용한 search 시작