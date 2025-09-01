### Create conda environment:
conda create -n aloha python=3.8.10
conda activate aloha
pip install -r requirements.txt

cd detr && pip install -e .

### Usages
activate conda environment:
conda activate aloha

### tranning:
bash train.sh

### evaluate:
1. source env
source Y1/devel/setup.bash
2. evaluate
bash eval_real_robot.sh