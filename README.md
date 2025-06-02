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
  - search.py > random search architecture 에 대한 score list를 구축하는 방식으로 변경
  - search.py > prepare_training()
  - search.py > compute_nas_score()
  - ./model/sam.py > Class Image encoder setting
  - ./model/mmseg/models/sam/image_encoder.py > Class Prompt generator setting
