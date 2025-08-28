### Create conda environment:
conda create -n aloha python=3.8.10
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython

cd detr && pip install -e .

### Usages
activate conda environment:
conda activate aloha

tranning:
python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir <ckpt dir> \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0

evaluate:
python3 eval_real_robot.py