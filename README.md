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
  - CAMO 데이터셋으로 실험 진행 중
  - 실험 돌아감 > loss랑 nas score 출력됨
  - sam-b: 15152M 정도 차지함